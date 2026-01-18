from __future__ import annotations
import logging # isort:skip
import sys
import types
import xyzservices
from bokeh.core.enums import enumeration
from .util.deprecation import deprecated
class _TileProvidersModule(types.ModuleType):

    def deprecated_vendors():
        deprecated((3, 0, 0), 'tile_providers module', 'add_tile directly')
        return enumeration('CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY', case_sensitive=True)
    Vendors = deprecated_vendors()

    def get_provider(self, provider_name: str | Vendors | xyzservices.TileProvider):
        """Use this function to retrieve an instance of a predefined tile provider.

        .. warning::
            get_provider is deprecated as of Bokeh 3.0.0 and will be removed in a future
            release. Use ``add_tile`` directly instead.

        Args:
            provider_name (Union[str, Vendors, xyzservices.TileProvider]):
                Name of the tile provider to supply.

                Use a ``tile_providers.Vendors`` enumeration value, or the string
                name of one of the known providers. Use
                :class:`xyzservices.TileProvider` to pass custom tile providers.

        Returns:
            WMTSTileProviderSource: The desired tile provider instance.

        Raises:
            ValueError: if the specified provider can not be found.

        Example:

            .. code-block:: python

                    >>> from bokeh.tile_providers import get_provider, Vendors
                    >>> get_provider(Vendors.CARTODBPOSITRON)
                    <class 'bokeh.models.tiles.WMTSTileSource'>
                    >>> get_provider('CARTODBPOSITRON')
                    <class 'bokeh.models.tiles.WMTSTileSource'>

                    >>> import xyzservices.providers as xyz
                    >>> get_provider(xyz.CartoDB.Positron)
                    <class 'bokeh.models.tiles.WMTSTileSource'>
        """
        deprecated((3, 0, 0), 'get_provider', 'add_tile directly')
        from bokeh.models import WMTSTileSource
        if isinstance(provider_name, WMTSTileSource):
            return WMTSTileSource(url=provider_name.url, attribution=provider_name.attribution)
        if isinstance(provider_name, str):
            provider_name = provider_name.lower()
            if provider_name == 'esri_imagery':
                provider_name = 'esri_worldimagery'
            if provider_name == 'osm':
                provider_name = 'openstreetmap_mapnik'
            if provider_name.startswith('stamen'):
                provider_name = f'stadia.{provider_name}'
            if 'retina' in provider_name:
                provider_name = provider_name.replace('retina', '')
                retina = True
            else:
                retina = False
            scale_factor = '@2x' if retina else None
            provider_name = xyzservices.providers.query_name(provider_name)
        else:
            scale_factor = None
        if isinstance(provider_name, xyzservices.TileProvider):
            return WMTSTileSource(url=provider_name.build_url(scale_factor=scale_factor), attribution=provider_name.html_attribution, min_zoom=provider_name.get('min_zoom', 0), max_zoom=provider_name.get('max_zoom', 30))
    CARTODBPOSITRON = Vendors.CARTODBPOSITRON
    CARTODBPOSITRON_RETINA = Vendors.CARTODBPOSITRON_RETINA
    STAMEN_TERRAIN = Vendors.STAMEN_TERRAIN
    STAMEN_TERRAIN_RETINA = Vendors.STAMEN_TERRAIN_RETINA
    STAMEN_TONER = Vendors.STAMEN_TONER
    STAMEN_TONER_BACKGROUND = Vendors.STAMEN_TONER_BACKGROUND
    STAMEN_TONER_LABELS = Vendors.STAMEN_TONER_LABELS
    OSM = Vendors.OSM
    ESRI_IMAGERY = Vendors.ESRI_IMAGERY