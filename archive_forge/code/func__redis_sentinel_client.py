from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def _redis_sentinel_client(redis_url: str, **kwargs: Any) -> RedisType:
    """helper method to parse an (un-official) redis+sentinel url
    and create a Sentinel connection to fetch the final redis client
    connection to a replica-master for read-write operations.

    If username and/or password for authentication is given the
    same credentials are used for the Redis Sentinel as well as Redis Server.
    With this implementation using a redis url only it is not possible
    to use different data for authentication on booth systems.
    """
    import redis
    parsed_url = urlparse(redis_url)
    sentinel_list = [(parsed_url.hostname or 'localhost', parsed_url.port or 26379)]
    if parsed_url.path:
        path_parts = parsed_url.path.split('/')
        service_name = path_parts[1] or 'mymaster'
        if len(path_parts) > 2:
            kwargs['db'] = path_parts[2]
    else:
        service_name = 'mymaster'
    sentinel_args = {}
    if parsed_url.password:
        sentinel_args['password'] = parsed_url.password
        kwargs['password'] = parsed_url.password
    if parsed_url.username:
        sentinel_args['username'] = parsed_url.username
        kwargs['username'] = parsed_url.username
    for arg in kwargs:
        if arg.startswith('ssl') or arg == 'client_name':
            sentinel_args[arg] = kwargs[arg]
    sentinel_client = redis.sentinel.Sentinel(sentinel_list, sentinel_kwargs=sentinel_args, **kwargs)
    try:
        sentinel_client.execute_command('ping')
    except redis.exceptions.AuthenticationError as ae:
        if 'no password is set' in ae.args[0]:
            logger.warning('Redis sentinel connection configured with password but Sentinel answered NO PASSWORD NEEDED - Please check Sentinel configuration')
            sentinel_client = redis.sentinel.Sentinel(sentinel_list, **kwargs)
        else:
            raise ae
    return sentinel_client.master_for(service_name)