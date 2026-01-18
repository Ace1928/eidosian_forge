from dash.development.base_component import Component, _explicitize_args
A LogoutButton component.
    Logout button to submit a form post request to the `logout_url` prop.
    Usage is intended for dash-deployment-server authentication.

    DDS usage:

    `dcc.LogoutButton(logout_url=os.getenv('DASH_LOGOUT_URL'))`

    Custom usage:

    - Implement a login mechanism.
    - Create a flask route with a post method handler.
    `@app.server.route('/logout', methods=['POST'])`
      - The logout route should perform what's necessary for the user to logout.
      - If you store the session in a cookie, clear the cookie:
      `rep = flask.Response(); rep.set_cookie('session', '', expires=0)`

    - Create a logout button component and assign it the logout_url
    `dcc.LogoutButton(logout_url='/logout')`

    See https://dash.plotly.com/dash-core-components/logout_button
    for more documentation and examples.

    Keyword arguments:

    - id (string; optional):
        Id of the button.

    - className (string; optional):
        CSS class for the button.

    - label (string; default 'Logout'):
        Text of the button.

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

    - logout_url (string; optional):
        Url to submit a post logout request.

    - method (string; default 'post'):
        Http method to submit the logout form.

    - style (dict; optional):
        Style of the button.