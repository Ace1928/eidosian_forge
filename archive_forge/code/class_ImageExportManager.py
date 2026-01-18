from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
class ImageExportManager(DockerBaseClass):

    def __init__(self, client):
        super(ImageExportManager, self).__init__()
        self.client = client
        parameters = self.client.module.params
        self.check_mode = self.client.check_mode
        self.path = parameters['path']
        self.force = parameters['force']
        self.tag = parameters['tag']
        if not is_valid_tag(self.tag, allow_empty=True):
            self.fail('"{0}" is not a valid docker tag'.format(self.tag))
        self.names = []
        for name in parameters['names']:
            if is_image_name_id(name):
                self.names.append({'id': name, 'joined': name})
            else:
                repo, repo_tag = parse_repository_tag(name)
                if not repo_tag:
                    repo_tag = self.tag
                self.names.append({'name': repo, 'tag': repo_tag, 'joined': '%s:%s' % (repo, repo_tag)})
        if not self.names:
            self.fail('At least one image name must be specified')

    def fail(self, msg):
        self.client.fail(msg)

    def get_export_reason(self):
        if self.force:
            return 'Exporting since force=true'
        try:
            archived_images = load_archived_image_manifest(self.path)
            if archived_images is None:
                return 'Overwriting since no image is present in archive'
        except ImageArchiveInvalidException as exc:
            self.log('Unable to extract manifest summary from archive: %s' % to_native(exc))
            return 'Overwriting an unreadable archive file'
        left_names = list(self.names)
        for archived_image in archived_images:
            found = False
            for i, name in enumerate(left_names):
                if name['id'] == api_image_id(archived_image.image_id) and [name['joined']] == archived_image.repo_tags:
                    del left_names[i]
                    found = True
                    break
            if not found:
                return 'Overwriting archive since it contains unexpected image %s named %s' % (archived_image.image_id, ', '.join(archived_image.repo_tags))
        if left_names:
            return 'Overwriting archive since it is missing image(s) %s' % ', '.join([name['joined'] for name in left_names])
        return None

    def write_chunks(self, chunks):
        try:
            with open(self.path, 'wb') as fd:
                for chunk in chunks:
                    fd.write(chunk)
        except Exception as exc:
            self.fail('Error writing image archive %s - %s' % (self.path, to_native(exc)))

    def export_images(self):
        image_names = [name['joined'] for name in self.names]
        image_names_str = ', '.join(image_names)
        if len(image_names) == 1:
            self.log('Getting archive of image %s' % image_names[0])
            try:
                chunks = self.client._stream_raw_result(self.client._get(self.client._url('/images/{0}/get', image_names[0]), stream=True), DEFAULT_DATA_CHUNK_SIZE, False)
            except Exception as exc:
                self.fail('Error getting image %s - %s' % (image_names[0], to_native(exc)))
        else:
            self.log('Getting archive of images %s' % image_names_str)
            try:
                chunks = self.client._stream_raw_result(self.client._get(self.client._url('/images/get'), stream=True, params={'names': image_names}), DEFAULT_DATA_CHUNK_SIZE, False)
            except Exception as exc:
                self.fail('Error getting images %s - %s' % (image_names_str, to_native(exc)))
        self.write_chunks(chunks)

    def run(self):
        tag = self.tag
        if not tag:
            tag = 'latest'
        images = []
        for name in self.names:
            if 'id' in name:
                image = self.client.find_image_by_id(name['id'], accept_missing_image=True)
            else:
                image = self.client.find_image(name=name['name'], tag=name['tag'])
            if not image:
                self.fail('Image %s not found' % name['joined'])
            images.append(image)
            name['id'] = image['Id']
        results = {'changed': False, 'images': images}
        reason = self.get_export_reason()
        if reason is not None:
            results['msg'] = reason
            results['changed'] = True
            if not self.check_mode:
                self.export_images()
        return results