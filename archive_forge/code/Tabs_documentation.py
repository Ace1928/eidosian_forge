from dash.development.base_component import Component, _explicitize_args
A Tabs component.
    A Dash component that lets you render pages with tabs - the Tabs component's children
    can be dcc.Tab components, which can hold a label that will be displayed as a tab, and can in turn hold
    children components that will be that tab's content.

    Keyword arguments:

    - children (list of a list of or a singular dash component, string or numbers | a list of or a singular dash component, string or number; optional):
        Array that holds Tab components.

    - id (string; optional):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - className (string; optional):
        Appends a class to the Tabs container holding the individual Tab
        components.

    - colors (dict; default {    border: '#d6d6d6',    primary: '#1975FA',    background: '#f9f9f9',}):
        Holds the colors used by the Tabs and Tab components. If you set
        these, you should specify colors for all properties, so: colors: {
        border: '#d6d6d6',    primary: '#1975FA',    background: '#f9f9f9'
        }.

        `colors` is a dict with keys:

        - background (string; optional)

        - border (string; optional)

        - primary (string; optional)

    - content_className (string; optional):
        Appends a class to the Tab content container holding the children
        of the Tab that is selected.

    - content_style (dict; optional):
        Appends (inline) styles to the tab content container holding the
        children of the Tab that is selected.

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

    - mobile_breakpoint (number; default 800):
        Breakpoint at which tabs are rendered full width (can be 0 if you
        don't want full width tabs on mobile).

    - parent_className (string; optional):
        Appends a class to the top-level parent container holding both the
        Tabs container and the content container.

    - parent_style (dict; optional):
        Appends (inline) styles to the top-level parent container holding
        both the Tabs container and the content container.

    - persisted_props (list of a value equal to: 'value's; default ['value']):
        Properties whose user interactions will persist after refreshing
        the component or the page. Since only `value` is allowed this prop
        can normally be ignored.

    - persistence (boolean | string | number; optional):
        Used to allow user interactions in this component to be persisted
        when the component - or the page - is refreshed. If `persisted` is
        truthy and hasn't changed from its previous value, a `value` that
        the user has changed while using the app will keep that change, as
        long as the new `value` also matches what was given originally.
        Used in conjunction with `persistence_type`.

    - persistence_type (a value equal to: 'local', 'session', 'memory'; default 'local'):
        Where persisted user changes will be stored: memory: only kept in
        memory, reset on page refresh. local: window.localStorage, data is
        kept after the browser quit. session: window.sessionStorage, data
        is cleared once the browser quit.

    - style (dict; optional):
        Appends (inline) styles to the Tabs container holding the
        individual Tab components.

    - value (string; optional):
        The value of the currently selected Tab.

    - vertical (boolean; default False):
        Renders the tabs vertically (on the side).