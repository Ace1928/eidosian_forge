from saharaclient.api import base
class ImageManagerV2(_ImageManager):

    def get_tags(self, image_id):
        return self._get('/images/%s/tags' % image_id)

    def update_tags(self, image_id, new_tags):
        return self._update('/images/%s/tags' % image_id, {'tags': new_tags})

    def delete_tags(self, image_id):
        return self._delete('/images/%s/tags' % image_id)