import json
import pkgutil
import textwrap
from typing import Callable, Dict, Optional, Tuple, Any, Union
import uuid
from ._vegafusion_data import compile_with_vegafusion, using_vegafusion
from .plugin_registry import PluginRegistry, PluginEnabler
from .mimebundle import spec_to_mimebundle
from .schemapi import validate_jsonschema
class RendererRegistry(PluginRegistry[RendererType]):
    entrypoint_err_messages = {'notebook': textwrap.dedent("\n            To use the 'notebook' renderer, you must install the vega package\n            and the associated Jupyter extension.\n            See https://altair-viz.github.io/getting_started/installation.html\n            for more information.\n            "), 'altair_viewer': textwrap.dedent("\n            To use the 'altair_viewer' renderer, you must install the altair_viewer\n            package; see http://github.com/altair-viz/altair_viewer/\n            for more information.\n            ")}

    def set_embed_options(self, defaultStyle: Optional[Union[bool, str]]=None, renderer: Optional[str]=None, width: Optional[int]=None, height: Optional[int]=None, padding: Optional[int]=None, scaleFactor: Optional[float]=None, actions: Optional[Union[bool, Dict[str, bool]]]=None, format_locale: Optional[Union[str, dict]]=None, time_format_locale: Optional[Union[str, dict]]=None, **kwargs) -> PluginEnabler:
        """Set options for embeddings of Vega & Vega-Lite charts.

        Options are fully documented at https://github.com/vega/vega-embed.
        Similar to the `enable()` method, this can be used as either
        a persistent global switch, or as a temporary local setting using
        a context manager (i.e. a `with` statement).

        Parameters
        ----------
        defaultStyle : bool or string
            Specify a default stylesheet for embed actions.
        renderer : string
            The renderer to use for the view. One of "canvas" (default) or "svg"
        width : integer
            The view width in pixels
        height : integer
            The view height in pixels
        padding : integer
            The view padding in pixels
        scaleFactor : number
            The number by which to multiply the width and height (default 1)
            of an exported PNG or SVG image.
        actions : bool or dict
            Determines if action links ("Export as PNG/SVG", "View Source",
            "View Vega" (only for Vega-Lite), "Open in Vega Editor") are
            included with the embedded view. If the value is true, all action
            links will be shown and none if the value is false. This property
            can take a key-value mapping object that maps keys (export, source,
            compiled, editor) to boolean values for determining if
            each action link should be shown.
        format_locale : str or dict
            d3-format locale name or dictionary. Defaults to "en-US" for United States English.
            See https://github.com/d3/d3-format/tree/main/locale for available names and example
            definitions.
        time_format_locale : str or dict
            d3-time-format locale name or dictionary. Defaults to "en-US" for United States English.
            See https://github.com/d3/d3-time-format/tree/main/locale for available names and example
            definitions.
        **kwargs :
            Additional options are passed directly to embed options.
        """
        options: Dict[str, Optional[Union[bool, str, float, Dict[str, bool]]]] = {'defaultStyle': defaultStyle, 'renderer': renderer, 'width': width, 'height': height, 'padding': padding, 'scaleFactor': scaleFactor, 'actions': actions, 'formatLocale': format_locale, 'timeFormatLocale': time_format_locale}
        kwargs.update({key: val for key, val in options.items() if val is not None})
        return self.enable(None, embed_options=kwargs)