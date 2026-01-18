from saharaclient.api import base
class _ImageManager(base.ResourceManager):
    resource_class = Image

    def list(self, search_opts=None):
        """Get a list of registered images."""
        query = base.get_query_string(search_opts)
        return self._list('/images%s' % query, 'images')

    def get(self, id):
        """Get information about an image"""
        return self._get('/images/%s' % id, 'image')

    def unregister_image(self, image_id):
        """Remove an Image from Sahara Image Registry."""
        self._delete('/images/%s' % image_id)

    def update_image(self, image_id, user_name, desc=None):
        """Create or update an Image in Image Registry."""
        desc = desc if desc else ''
        data = {'username': user_name, 'description': desc}
        return self._post('/images/%s' % image_id, data)