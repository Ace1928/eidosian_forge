from dash.development.base_component import Component, _explicitize_args
A Markdown component.
    A component that renders Markdown text as specified by the
    GitHub Markdown spec. These component uses
    [react-markdown](https://rexxars.github.io/react-markdown/) under the hood.

    Keyword arguments:

    - children (string | list of strings; optional):
        A markdown string (or array of strings) that adheres to the
        CommonMark spec.

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - className (string; optional):
        Class name of the container element.

    - dangerously_allow_html (boolean; default False):
        A boolean to control raw HTML escaping. Setting HTML from code is
        risky because it's easy to inadvertently expose your users to a
        cross-site scripting (XSS)
        (https://en.wikipedia.org/wiki/Cross-site_scripting) attack.

    - dedent (boolean; default True):
        Remove matching leading whitespace from all lines. Lines that are
        empty, or contain *only* whitespace, are ignored. Both spaces and
        tab characters are removed, but only if they match; we will not
        convert tabs to spaces or vice versa.

    - highlight_config (dict; optional):
        Config options for syntax highlighting.

        `highlight_config` is a dict with keys:

        - theme (a value equal to: 'dark', 'light'; optional):
            Color scheme; default 'light'.

    - link_target (string; optional):
        A string for the target attribute to use on links (such as
        "_blank").

    - loading_state (dict; optional):
        Object that holds the loading state object coming from
        dash-renderer.

        `loading_state` is a dict with keys:

        - component_name (string; optional):
            Holds the name of the component that is loading.

        - is_loading (boolean; optional):
            Determines if the component is loading or not.

        - prop_name (string; optional):
            Holds which property is loading.

    - mathjax (boolean; default False):
        If True, loads mathjax v3 (tex-svg) into the page and use it in
        the markdown.

    - style (dict; optional):
        User-defined inline styles for the rendered Markdown.