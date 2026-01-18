import logging
import os
import appdirs
import dogpile.cache
Retrieve the version stored for an ironic 'host', if it's not stale.

    Check to see if there is valid cached data for the host/port
    combination and return that if it isn't stale.

    param host: The host that we need to retrieve data for
    param port: The port on the host that we need to retrieve data for
    param expiry: The age in seconds before cached data is deemed invalid
    