from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
class RegistryManager(base.Manager):
    resource_class = Registry

    @staticmethod
    def _path(id=None):
        if id:
            return '/v1/registries/%s' % id
        else:
            return '/v1/registries'

    def list(self, marker=None, limit=None, sort_key=None, sort_dir=None, all_projects=False, **kwargs):
        """Retrieve a list of registries.

        :param all_projects: Optional, list registries in all projects

        :param marker: Optional, the UUID of a registries, eg the last
                       registries from a previous result set. Return
                       the next result set.
        :param limit: The maximum number of results to return per
                      request, if:

            1) limit > 0, the maximum number of registries to return.
            2) limit param is NOT specified (None), the number of items
               returned respect the maximum imposed by the ZUN API
               (see Zun's api.max_limit option).

        :param sort_key: Optional, field used for sorting.

        :param sort_dir: Optional, direction of sorting, either 'asc' (the
                         default) or 'desc'.

        :returns: A list of registries.

        """
        if limit is not None:
            limit = int(limit)
        filters = utils.common_filters(marker, limit, sort_key, sort_dir, all_projects)
        path = ''
        if filters:
            path += '?' + '&'.join(filters)
        if limit is None:
            return self._list(self._path(path), 'registries', qparams=kwargs)
        else:
            return self._list_pagination(self._path(path), 'registries', limit=limit)

    def get(self, id, **kwargs):
        try:
            return self._list(self._path(id), qparams=kwargs)[0]
        except IndexError:
            return None

    def create(self, **kwargs):
        new = {'registry': {}}
        for key, value in kwargs.items():
            if key in CREATION_ATTRIBUTES:
                new['registry'][key] = value
            else:
                raise exceptions.InvalidAttribute('Key must be in %s' % ','.join(CREATION_ATTRIBUTES))
        return self._create(self._path(), new)

    def delete(self, id, **kwargs):
        return self._delete(self._path(id), qparams=kwargs)

    def update(self, id, **patch):
        kwargs = {'registry': patch}
        return self._update(self._path(id), kwargs)