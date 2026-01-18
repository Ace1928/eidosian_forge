from dash.development.base_component import Component, _explicitize_args
A Clipboard component.
    The Clipboard component copies text to the clipboard

    Keyword arguments:

    - id (string; optional):
        The ID used to identify this component.

    - className (string; optional):
        The class  name of the icon element.

    - content (string; optional):
        The text to be copied to the clipboard if the `target_id` is None.

    - html_content (string; optional):
        The clipboard html text be copied to the clipboard if the
        `target_id` is None.

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

    - n_clicks (number; default 0):
        The number of times copy button was clicked.

    - style (dict; optional):
        The icon's styles.

    - target_id (string | dict; optional):
        The id of target component containing text to copy to the
        clipboard. The inner text of the `children` prop will be copied to
        the clipboard.  If none, then the text from the  `value` prop will
        be copied.

    - title (string; optional):
        The text shown as a tooltip when hovering over the copy icon.