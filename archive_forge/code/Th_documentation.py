from dash.development.base_component import Component, _explicitize_args
A Th component.
    Th is a wrapper for the <th> HTML5 element.
    For detailed attribute info see:
    https://developer.mozilla.org/en-US/docs/Web/HTML/Element/th

    Keyword arguments:

    - children (a list of or a singular dash component, string or number; optional):
        The children of this component.

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - accessKey (string; optional):
        Keyboard shortcut to activate or add focus to the element.

    - aria-* (string; optional):
        A wildcard aria attribute.

    - className (string; optional):
        Often used with CSS to style elements with common properties.

    - colSpan (string | number; optional):
        The colspan attribute defines the number of columns a cell should
        span.

    - contentEditable (string; optional):
        Indicates whether the element's content is editable.

    - data-* (string; optional):
        A wildcard data attribute.

    - dir (string; optional):
        Defines the text direction. Allowed values are ltr (Left-To-Right)
        or rtl (Right-To-Left).

    - disable_n_clicks (boolean; optional):
        When True, this will disable the n_clicks prop.  Use this to
        remove event listeners that may interfere with screen readers.

    - draggable (string; optional):
        Defines whether the element can be dragged.

    - headers (string; optional):
        IDs of the <th> elements which applies to this element.

    - hidden (a value equal to: 'hidden', 'HIDDEN' | boolean; optional):
        Prevents rendering of given element, while keeping child elements,
        e.g. script elements, active.

    - key (string; optional):
        A unique identifier for the component, used to improve performance
        by React.js while rendering components See
        https://reactjs.org/docs/lists-and-keys.html for more info.

    - lang (string; optional):
        Defines the language used in the element.

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
        An integer that represents the number of times that this element
        has been clicked on.

    - n_clicks_timestamp (number; default -1):
        An integer that represents the time (in ms since 1970) at which
        n_clicks changed. This can be used to tell which button was
        changed most recently.

    - role (string; optional):
        Defines an explicit role for an element for use by assistive
        technologies.

    - rowSpan (string | number; optional):
        Defines the number of rows a table cell should span over.

    - scope (string; optional):
        Defines the cells that the header test (defined in the th element)
        relates to.

    - spellCheck (string; optional):
        Indicates whether spell checking is allowed for the element.

    - style (dict; optional):
        Defines CSS styles which will override styles previously set.

    - tabIndex (string; optional):
        Overrides the browser's default tab order and follows the one
        specified instead.

    - title (string; optional):
        Text to be displayed in a tooltip when hovering over the element.