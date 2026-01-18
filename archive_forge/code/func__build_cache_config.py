import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
def _build_cache_config(conf):
    """Build the cache region dictionary configuration.

    :returns: dict
    """
    prefix = conf.cache.config_prefix
    conf_dict = {}
    conf_dict['%s.backend' % prefix] = _opts._DEFAULT_BACKEND
    if conf.cache.enabled is True:
        conf_dict['%s.backend' % prefix] = conf.cache.backend
    conf_dict['%s.expiration_time' % prefix] = conf.cache.expiration_time
    for argument in conf.cache.backend_argument:
        try:
            argname, argvalue = argument.split(':', 1)
        except ValueError:
            msg = 'Unable to build cache config-key. Expected format "<argname>:<value>". Skipping unknown format: %s'
            _LOG.error(msg, argument)
            continue
        arg_key = '.'.join([prefix, 'arguments', argname])
        if conf.cache.backend in ('dogpile.cache.memcached', 'oslo_cache.memcache_pool') and argname == 'url':
            argvalue = argvalue.split(',')
        conf_dict[arg_key] = argvalue
        _LOG.debug('Oslo Cache Config: %s', conf_dict)
    if conf.cache.backend == 'dogpile.cache.redis':
        if conf.cache.redis_password is None:
            netloc = conf.cache.redis_server
        elif conf.cache.redis_username:
            netloc = '%s:%s@%s' % (conf.cache.redis_username, conf.cache.redis_password, conf.cache.redis_server)
        else:
            netloc = ':%s@%s' % (conf.cache.redis_password, conf.cache.redis_server)
        parts = urllib.parse.ParseResult(scheme='rediss' if conf.cache.tls_enabled else 'redis', netloc=netloc, path='', params='', query='', fragment='')
        conf_dict.setdefault('%s.arguments.url' % prefix, urllib.parse.urlunparse(parts))
        for arg in ('socket_timeout',):
            value = getattr(conf.cache, 'redis_' + arg)
            conf_dict['%s.arguments.%s' % (prefix, arg)] = value
    elif conf.cache.backend == 'dogpile.cache.redis_sentinel':
        for arg in ('password', 'socket_timeout'):
            value = getattr(conf.cache, 'redis_' + arg)
            conf_dict['%s.arguments.%s' % (prefix, arg)] = value
        if conf.cache.redis_username:
            conf_dict['%s.arguments.connection_kwargs' % prefix] = {'username': conf.cache.redis_username}
            conf_dict['%s.arguments.sentinel_kwargs' % prefix] = {'username': conf.cache.redis_username}
        conf_dict['%s.arguments.service_name' % prefix] = conf.cache.redis_sentinel_service_name
        if conf.cache.redis_sentinels:
            conf_dict['%s.arguments.sentinels' % prefix] = [_parse_sentinel(s) for s in conf.cache.redis_sentinels]
    else:
        conf_dict.setdefault('%s.arguments.url' % prefix, conf.cache.memcache_servers)
        for arg in ('dead_retry', 'socket_timeout', 'pool_maxsize', 'pool_unused_timeout', 'pool_connection_get_timeout', 'pool_flush_on_reconnect', 'sasl_enabled', 'username', 'password'):
            value = getattr(conf.cache, 'memcache_' + arg)
            conf_dict['%s.arguments.%s' % (prefix, arg)] = value
    if conf.cache.tls_enabled:
        if conf.cache.backend in ('dogpile.cache.bmemcache', 'dogpile.cache.pymemcache', 'oslo_cache.memcache_pool'):
            _LOG.debug('Oslo Cache TLS - CA: %s', conf.cache.tls_cafile)
            tls_context = ssl.create_default_context(cafile=conf.cache.tls_cafile)
            if conf.cache.enforce_fips_mode:
                if hasattr(ssl, 'FIPS_mode'):
                    _LOG.info('Enforcing the use of the OpenSSL FIPS mode')
                    ssl.FIPS_mode_set(1)
                else:
                    raise exception.ConfigurationError("OpenSSL FIPS mode is not supported by your Python version. You must either change the Python executable used to a version with FIPS mode support or disable FIPS mode by setting the '[cache] enforce_fips_mode' configuration option to 'False'.")
            if conf.cache.tls_certfile is not None:
                _LOG.debug('Oslo Cache TLS - cert: %s', conf.cache.tls_certfile)
                _LOG.debug('Oslo Cache TLS - key: %s', conf.cache.tls_keyfile)
                tls_context.load_cert_chain(conf.cache.tls_certfile, conf.cache.tls_keyfile)
            if conf.cache.tls_allowed_ciphers is not None:
                _LOG.debug('Oslo Cache TLS - ciphers: %s', conf.cache.tls_allowed_ciphers)
                tls_context.set_ciphers(conf.cache.tls_allowed_ciphers)
            conf_dict['%s.arguments.tls_context' % prefix] = tls_context
        elif conf.cache.backend in ('dogpile.cache.redis', 'dogpile.cache.redis_sentinel'):
            if conf.cache.tls_allowed_ciphers is not None:
                raise exception.ConfigurationError('Limiting allowed ciphers is not supported by the %s backend' % conf.cache.backend)
            if conf.cache.enforce_fips_mode:
                raise exception.ConfigurationError('FIPS mode is not supported by the %s backend' % conf.cache.backend)
            conn_kwargs = {}
            if conf.cache.tls_cafile is not None:
                _LOG.debug('Oslo Cache TLS - CA: %s', conf.cache.tls_cafile)
                conn_kwargs['ssl_ca_certs'] = conf.cache.tls_cafile
            if conf.cache.tls_certfile is not None:
                _LOG.debug('Oslo Cache TLS - cert: %s', conf.cache.tls_certfile)
                _LOG.debug('Oslo Cache TLS - key: %s', conf.cache.tls_keyfile)
                conn_kwargs.update({'ssl_certfile': conf.cache.tls_certfile, 'ssl_keyfile': conf.cache.tls_keyfile})
            if conf.cache.backend == 'dogpile.cache.redis_sentinel':
                conn_kwargs.update({'ssl': True})
                conf_dict.setdefault('%s.arguments.connection_kwargs' % prefix, {}).update(conn_kwargs)
                conf_dict.setdefault('%s.arguments.sentinel_kwargs' % prefix, {}).update(conn_kwargs)
            else:
                conf_dict.setdefault('%s.arguments.connection_kwargs' % prefix, {}).update(conn_kwargs)
        else:
            raise exception.ConfigurationError('TLS setting via [cache] tls_enabled is not supported by the %s backend. Set [cache] tls_enabled=False or use a different backend.' % conf.cache.backend)
    if conf.cache.enable_socket_keepalive:
        if conf.cache.backend != 'dogpile.cache.pymemcache':
            msg = _("Socket keepalive is only supported by the 'dogpile.cache.pymemcache' backend.")
            raise exception.ConfigurationError(msg)
        import pymemcache
        socket_keepalive = pymemcache.KeepaliveOpts(idle=conf.cache.socket_keepalive_idle, intvl=conf.cache.socket_keepalive_interval, cnt=conf.cache.socket_keepalive_count)
        conf_dict['%s.arguments.socket_keepalive' % prefix] = socket_keepalive
    if conf.cache.enable_retry_client:
        if conf.cache.backend != 'dogpile.cache.pymemcache':
            msg = _("Retry client is only supported by the 'dogpile.cache.pymemcache' backend.")
            raise exception.ConfigurationError(msg)
        import pymemcache
        conf_dict['%s.arguments.enable_retry_client' % prefix] = True
        conf_dict['%s.arguments.retry_attempts' % prefix] = conf.cache.retry_attempts
        conf_dict['%s.arguments.retry_delay' % prefix] = conf.cache.retry_delay
        conf_dict['%s.arguments.hashclient_retry_attempts' % prefix] = conf.cache.hashclient_retry_attempts
        conf_dict['%s.arguments.hashclient_retry_delay' % prefix] = conf.cache.hashclient_retry_delay
        conf_dict['%s.arguments.dead_timeout' % prefix] = conf.cache.dead_timeout
    return conf_dict