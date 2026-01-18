import logging
import os
import time
from . import docker_base as base
def _create_config_os_ken_bgp(self):
    c = base.CmdBuffer()
    c << 'import os'
    c << ''
    c << 'BGP = {'
    c << "    'local_as': %s," % str(self.asn)
    c << "    'router_id': '%s'," % self.router_id
    c << "    'neighbors': ["
    c << '        {'
    for peer, info in self.peers.items():
        n_addr = info['neigh_addr'].split('/')[0]
        c << "            'address': '%s'," % n_addr
        c << "            'remote_as': %s," % str(peer.asn)
        c << "            'enable_ipv4': True,"
        c << "            'enable_ipv6': True,"
        c << "            'enable_vpnv4': True,"
        c << "            'enable_vpnv6': True,"
        c << '        },'
        c << '    ],'
    c << "    'routes': ["
    for route in self.routes.values():
        c << '        {'
        c << "            'prefix': '%s'," % route['prefix']
        c << '        },'
    c << '    ],'
    c << '}'
    log_conf = "LOGGING = {\n\n    # We use python logging package for logging.\n    'version': 1,\n    'disable_existing_loggers': False,\n\n    'formatters': {\n        'verbose': {\n            'format': '%(levelname)s %(asctime)s %(module)s ' +\n                      '[%(process)d %(thread)d] %(message)s'\n        },\n        'simple': {\n            'format': '%(levelname)s %(asctime)s %(module)s %(lineno)s ' +\n                      '%(message)s'\n        },\n        'stats': {\n            'format': '%(message)s'\n        },\n    },\n\n    'handlers': {\n        # Outputs log to console.\n        'console': {\n            'level': 'DEBUG',\n            'class': 'logging.StreamHandler',\n            'formatter': 'simple'\n        },\n        'console_stats': {\n            'level': 'DEBUG',\n            'class': 'logging.StreamHandler',\n            'formatter': 'stats'\n        },\n        # Rotates log file when its size reaches 10MB.\n        'log_file': {\n            'level': 'DEBUG',\n            'class': 'logging.handlers.RotatingFileHandler',\n            'filename': os.path.join('.', 'bgpspeaker.log'),\n            'maxBytes': '10000000',\n            'formatter': 'verbose'\n        },\n        'stats_file': {\n            'level': 'DEBUG',\n            'class': 'logging.handlers.RotatingFileHandler',\n            'filename': os.path.join('.', 'statistics_bgps.log'),\n            'maxBytes': '10000000',\n            'formatter': 'stats'\n        },\n    },\n\n    # Fine-grained control of logging per instance.\n    'loggers': {\n        'bgpspeaker': {\n            'handlers': ['console', 'log_file'],\n            'handlers': ['console'],\n            'level': 'DEBUG',\n            'propagate': False,\n        },\n        'stats': {\n            'handlers': ['stats_file', 'console_stats'],\n            'level': 'INFO',\n            'propagate': False,\n            'formatter': 'stats',\n        },\n    },\n\n    # Root loggers.\n    'root': {\n        'handlers': ['console', 'log_file'],\n        'level': 'DEBUG',\n        'propagate': True,\n    },\n}"
    c << log_conf
    with open(os.path.join(self.config_dir, 'bgp_conf.py'), 'w') as f:
        LOG.info("[%s's new config]", self.name)
        LOG.info(str(c))
        f.writelines(str(c))