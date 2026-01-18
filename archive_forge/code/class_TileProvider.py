from __future__ import annotations
import json
import urllib.request
import uuid
from typing import Callable
from urllib.parse import quote
class TileProvider(Bunch):
    """
    A dict with attribute-access and that
    can be called to update keys


    Examples
    --------

    You can create custom :class:`TileProvider` by passing your attributes to the object
    as it would have been a ``dict()``. It is required to always specify ``name``,
    ``url``, and ``attribution``.

    >>> public_provider = TileProvider(
    ...     name="My public tiles",
    ...     url="https://myserver.com/tiles/{z}/{x}/{y}.png",
    ...     attribution="(C) xyzservices",
    ... )

    Alternatively, you can create it from a dictionary of attributes. When specifying a
    placeholder for the access token, please use the ``"<insert your access token
    here>"`` string to ensure that :meth:`~xyzservices.TileProvider.requires_token`
    method works properly.

    >>> private_provider = TileProvider(
    ...    {
    ...        "url": "https://myserver.com/tiles/{z}/{x}/{y}.png?apikey={accessToken}",
    ...        "attribution": "(C) xyzservices",
    ...        "accessToken": "<insert your access token here>",
    ...        "name": "my_private_provider",
    ...    }
    ... )

    It is customary to include ``html_attribution`` attribute containing HTML string as
    ``'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>
    contributors'`` alongisde a plain-text ``attribution``.

    You can then fetch all information as attributes:

    >>> public_provider.url
    'https://myserver.com/tiles/{z}/{x}/{y}.png'

    >>> public_provider.attribution
    '(C) xyzservices'

    To ensure you will be able to use the tiles, you can check if the
    :class:`TileProvider` requires a token or API key.

    >>> public_provider.requires_token()
    False
    >>> private_provider.requires_token()
    True

    You can also generate URL in the required format with or without placeholders:

    >>> public_provider.build_url()
    'https://myserver.com/tiles/{z}/{x}/{y}.png'
    >>> private_provider.build_url(x=12, y=21, z=11, accessToken="my_token")
    'https://myserver.com/tiles/11/12/21.png?access_token=my_token'

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        missing = []
        for el in ['name', 'url', 'attribution']:
            if el not in self.keys():
                missing.append(el)
        if len(missing) > 0:
            msg = f'The attributes `name`, `url`, and `attribution` are required to initialise a `TileProvider`. Please provide values for: `{'`, `'.join(missing)}`'
            raise AttributeError(msg)

    def __call__(self, **kwargs) -> TileProvider:
        new = TileProvider(self)
        new.update(kwargs)
        return new

    def copy(self) -> TileProvider:
        new = TileProvider(self)
        return new

    def build_url(self, x: int | str | None=None, y: int | str | None=None, z: int | str | None=None, scale_factor: str | None=None, fill_subdomain: bool | None=True, **kwargs) -> str:
        """
        Build the URL of tiles from the :class:`TileProvider` object

        Can return URL with placeholders or the final tile URL.

        Parameters
        ----------

        x, y, z : int (optional)
            tile number
        scale_factor : str (optional)
            Scale factor (where supported). For example, you can get double resolution
            (512 x 512) instead of standard one (256 x 256) with ``"@2x"``. If you want
            to keep a placeholder, pass `"{r}"`.
        fill_subdomain : bool (optional, default True)
            Fill subdomain placeholder with the first available subdomain. If False, the
            URL will contain ``{s}`` placeholder for subdomain.

        **kwargs
            Other potential attributes updating the :class:`TileProvider`.

        Returns
        -------

        url : str
            Formatted URL

        Examples
        --------
        >>> import xyzservices.providers as xyz

        >>> xyz.CartoDB.DarkMatter.build_url()
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'

        >>> xyz.CartoDB.DarkMatter.build_url(x=9, y=11, z=5)
        'https://a.basemaps.cartocdn.com/dark_all/5/9/11.png'

        >>> xyz.CartoDB.DarkMatter.build_url(x=9, y=11, z=5, scale_factor="@2x")
        'https://a.basemaps.cartocdn.com/dark_all/5/9/11@2x.png'

        >>> xyz.MapBox.build_url(accessToken="my_token")
        'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=my_token'

        """
        provider = self.copy()
        if x is None:
            x = '{x}'
        if y is None:
            y = '{y}'
        if z is None:
            z = '{z}'
        provider.update(kwargs)
        if provider.requires_token():
            raise ValueError('Token is required for this provider, but not provided. You can either update TileProvider or pass respective keywords to build_url().')
        url = provider.pop('url')
        if scale_factor:
            r = scale_factor
            provider.pop('r', None)
        else:
            r = provider.pop('r', '')
        if fill_subdomain:
            subdomains = provider.pop('subdomains', 'abc')
            s = subdomains[0]
        else:
            s = '{s}'
        return url.format(x=x, y=y, z=z, s=s, r=r, **provider)

    def requires_token(self) -> bool:
        """
        Returns ``True`` if the TileProvider requires access token to fetch tiles.

        The token attribute name vary and some :class:`TileProvider` objects may require
        more than one token (e.g. ``HERE``). The information is deduced from the
        presence of `'<insert your...'` string in one or more of attributes. When
        specifying a placeholder for the access token, please use the ``"<insert your
        access token here>"`` string to ensure that
        :meth:`~xyzservices.TileProvider.requires_token` method works properly.

        Returns
        -------
        bool

        Examples
        --------
        >>> import xyzservices.providers as xyz
        >>> xyz.MapBox.requires_token()
        True

        >>> xyz.CartoDB.Positron
        False

        We can specify this API key by calling the object or overriding the attribute.
        Overriding the attribute will alter existing object:

        >>> xyz.OpenWeatherMap.Clouds["apiKey"] = "my-private-api-key"

        Calling the object will return a copy:

        >>> xyz.OpenWeatherMap.Clouds(apiKey="my-private-api-key")


        """
        for key, val in self.items():
            if isinstance(val, str) and '<insert your' in val and (key in self.url):
                return True
        return False

    @property
    def html_attribution(self):
        if 'html_attribution' in self:
            return self['html_attribution']
        return self['attribution']

    def _repr_html_(self, inside=False):
        provider_info = ''
        for key, val in self.items():
            if key != 'name':
                provider_info += f'<dt><span>{key}</span></dt><dd>{val}</dd>'
        style = '' if inside else f'<style>{CSS_STYLE}</style>'
        html = f'\n        <div>\n        {style}\n            <div class="xyz-wrap">\n                <div class="xyz-header">\n                    <div class="xyz-obj">xyzservices.TileProvider</div>\n                    <div class="xyz-name">{self.name}</div>\n                </div>\n                <div class="xyz-details">\n                    <dl class="xyz-attrs">\n                        {provider_info}\n                    </dl>\n                </div>\n            </div>\n        </div>\n        '
        return html

    @classmethod
    def from_qms(cls, name: str) -> TileProvider:
        """
        Creates a :class:`TileProvider` object based on the definition from
        the `Quick Map Services <https://qms.nextgis.com/>`__ open catalog.

        Parameters
        ----------
        name : str
            Service name

        Returns
        -------
        :class:`TileProvider`

        Examples
        --------
        >>> from xyzservices.lib import TileProvider
        >>> provider = TileProvider.from_qms("OpenTopoMap")
        """
        qms_api_url = 'https://qms.nextgis.com/api/v1/geoservices'
        services = json.load(urllib.request.urlopen(f'{qms_api_url}/?search={quote(name)}&type=tms'))
        for service in services:
            if service['name'] == name:
                break
        else:
            raise ValueError(f"Service '{name}' not found.")
        service_id = service['id']
        service_details = json.load(urllib.request.urlopen(f'{qms_api_url}/{service_id}'))
        return cls(name=service_details['name'], url=service_details['url'], min_zoom=service_details.get('z_min'), max_zoom=service_details.get('z_max'), attribution=service_details.get('copyright_text'))