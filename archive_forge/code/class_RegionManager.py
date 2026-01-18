from debtcollector import removals
from keystoneclient import base
class RegionManager(base.CrudManager):
    """Manager class for manipulating Identity regions."""
    resource_class = Region
    collection_key = 'regions'
    key = 'region'

    @removals.removed_kwarg('enabled', message='The enabled parameter is deprecated.', version='3.18.0', removal_version='4.0.0')
    def create(self, id=None, description=None, enabled=True, parent_region=None, **kwargs):
        """Create a region.

        :param str id: the unique identifier of the region. If not specified an
                       ID will be created by the server.
        :param str description: the description of the region.
        :param bool enabled: whether the region is enabled or not, determining
                             if it appears in the catalog.
        :param parent_region: the parent of the region in the hierarchy.
        :type parent_region: str or :class:`keystoneclient.v3.regions.Region`
        :param kwargs: any other attribute provided will be passed to the
                       server.

        :returns: the created region returned from server.
        :rtype: :class:`keystoneclient.v3.regions.Region`

        """
        return super(RegionManager, self).create(id=id, description=description, enabled=enabled, parent_region_id=base.getid(parent_region), **kwargs)

    def get(self, region):
        """Retrieve a region.

        :param region: the region to be retrieved from the server.
        :type region: str or :class:`keystoneclient.v3.regions.Region`

        :returns: the specified region returned from server.
        :rtype: :class:`keystoneclient.v3.regions.Region`

        """
        return super(RegionManager, self).get(region_id=base.getid(region))

    def list(self, **kwargs):
        """List regions.

        :param kwargs: any attributes provided will filter regions on.

        :returns: a list of regions.
        :rtype: list of :class:`keystoneclient.v3.regions.Region`.

        """
        return super(RegionManager, self).list(**kwargs)

    @removals.removed_kwarg('enabled', message='The enabled parameter is deprecated.', version='3.18.0', removal_version='4.0.0')
    def update(self, region, description=None, enabled=None, parent_region=None, **kwargs):
        """Update a region.

        :param region: the region to be updated on the server.
        :type region: str or :class:`keystoneclient.v3.regions.Region`
        :param str description: the new description of the region.
        :param bool enabled: determining if the region appears in the catalog
                             by enabling or disabling it.
        :param parent_region: the new parent of the region in the hierarchy.
        :type parent_region: str or :class:`keystoneclient.v3.regions.Region`
        :param kwargs: any other attribute provided will be passed to server.

        :returns: the updated region returned from server.
        :rtype: :class:`keystoneclient.v3.regions.Region`

        """
        return super(RegionManager, self).update(region_id=base.getid(region), description=description, enabled=enabled, parent_region_id=base.getid(parent_region), **kwargs)

    def delete(self, region):
        """Delete a region.

        :param region: the region to be deleted on the server.
        :type region: str or :class:`keystoneclient.v3.regions.Region`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
        return super(RegionManager, self).delete(region_id=base.getid(region))