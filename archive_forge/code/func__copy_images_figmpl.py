from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib
def _copy_images_figmpl(self, node):
    if node['srcset']:
        srcset = _parse_srcsetNodes(node['srcset'])
    else:
        srcset = None
    docsource = PurePath(self.document['source']).parent
    srctop = self.builder.srcdir
    rel = relpath(docsource, srctop).replace('.', '').replace(os.sep, '-')
    if len(rel):
        rel += '-'
    imagedir = PurePath(self.builder.outdir, self.builder.imagedir)
    Path(imagedir).mkdir(parents=True, exist_ok=True)
    if srcset:
        for src in srcset.values():
            abspath = PurePath(docsource, src)
            name = rel + abspath.name
            shutil.copyfile(abspath, imagedir / name)
    else:
        abspath = PurePath(docsource, node['uri'])
        name = rel + abspath.name
        shutil.copyfile(abspath, imagedir / name)
    return (imagedir, srcset, rel)