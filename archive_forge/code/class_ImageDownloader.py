import os
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import epoch_to_rfc1123, logging, requests, rfc1123_to_epoch, sha1
from sphinx.util.images import get_image_extension, guess_mimetype, parse_data_uri
from sphinx.util.osutil import ensuredir
class ImageDownloader(BaseImageConverter):
    default_priority = 100

    def match(self, node: nodes.image) -> bool:
        if self.app.builder.supported_image_types == []:
            return False
        elif self.app.builder.supported_remote_images:
            return False
        else:
            return '://' in node['uri']

    def handle(self, node: nodes.image) -> None:
        try:
            basename = os.path.basename(node['uri'])
            if '?' in basename:
                basename = basename.split('?')[0]
            if basename == '' or len(basename) > MAX_FILENAME_LEN:
                filename, ext = os.path.splitext(node['uri'])
                basename = sha1(filename.encode()).hexdigest() + ext
            basename = re.sub(CRITICAL_PATH_CHAR_RE, '_', basename)
            dirname = node['uri'].replace('://', '/').translate({ord('?'): '/', ord('&'): '/'})
            if len(dirname) > MAX_FILENAME_LEN:
                dirname = sha1(dirname.encode()).hexdigest()
            ensuredir(os.path.join(self.imagedir, dirname))
            path = os.path.join(self.imagedir, dirname, basename)
            headers = {}
            if os.path.exists(path):
                timestamp: float = ceil(os.stat(path).st_mtime)
                headers['If-Modified-Since'] = epoch_to_rfc1123(timestamp)
            r = requests.get(node['uri'], headers=headers)
            if r.status_code >= 400:
                logger.warning(__('Could not fetch remote image: %s [%d]') % (node['uri'], r.status_code))
            else:
                self.app.env.original_image_uri[path] = node['uri']
                if r.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(r.content)
                last_modified = r.headers.get('last-modified')
                if last_modified:
                    timestamp = rfc1123_to_epoch(last_modified)
                    os.utime(path, (timestamp, timestamp))
                mimetype = guess_mimetype(path, default='*')
                if mimetype != '*' and os.path.splitext(basename)[1] == '':
                    ext = get_image_extension(mimetype)
                    newpath = os.path.join(self.imagedir, dirname, basename + ext)
                    os.replace(path, newpath)
                    self.app.env.original_image_uri.pop(path)
                    self.app.env.original_image_uri[newpath] = node['uri']
                    path = newpath
                node['candidates'].pop('?')
                node['candidates'][mimetype] = path
                node['uri'] = path
                self.app.env.images.add_file(self.env.docname, path)
        except Exception as exc:
            logger.warning(__('Could not fetch remote image: %s [%s]') % (node['uri'], exc))