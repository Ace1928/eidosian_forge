from lazr.restfulclient.resource import (
class RecipeSet(CollectionWithKeyBasedLookup):
    """A custom subclass capable of recipe lookup by recipe ID."""

    def _get_url_from_id(self, id):
        """Transform a recipe ID into the URL to a recipe resource."""
        return str(self._root._root_uri.ensureSlash()) + 'recipes/' + str(id)
    collection_of = 'recipe'