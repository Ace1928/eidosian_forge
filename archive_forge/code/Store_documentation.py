from dash.development.base_component import Component, _explicitize_args
A Store component.
    Easily keep data on the client side with this component.
    The data is not inserted in the DOM.
    Data can be in memory, localStorage or sessionStorage.
    The data will be kept with the id as key.

    Keyword arguments:

    - id (string; required):
        The ID of this component, used to identify dash components in
        callbacks. The ID needs to be unique across all of the components
        in an app.

    - clear_data (boolean; default False):
        Set to True to remove the data contained in `data_key`.

    - data (dict | list | number | string | boolean; optional):
        The stored data for the id.

    - modified_timestamp (number; default -1):
        The last time the storage was modified.

    - storage_type (a value equal to: 'local', 'session', 'memory'; default 'memory'):
        The type of the web storage.  memory: only kept in memory, reset
        on page refresh. local: window.localStorage, data is kept after
        the browser quit. session: window.sessionStorage, data is cleared
        once the browser quit.