import argparse
import contextlib
import io
import json
import logging
import os
import pkgutil
import sys
from apitools.base.py import exceptions
from apitools.gen import gen_client_lib
from apitools.gen import util
def _GetCodegenFromFlags(args):
    """Create a codegen object from flags."""
    discovery_doc = _GetDiscoveryDocFromFlags(args)
    names = util.Names(args.strip_prefix, args.experimental_name_convention, args.experimental_capitalize_enums)
    if args.client_json:
        try:
            with io.open(args.client_json, encoding='utf8') as client_json:
                f = json.loads(util.ReplaceHomoglyphs(client_json.read()))
                web = f.get('installed', f.get('web', {}))
                client_id = web.get('client_id')
                client_secret = web.get('client_secret')
        except IOError:
            raise exceptions.NotFoundError('Failed to open client json file: %s' % args.client_json)
    else:
        client_id = args.client_id
        client_secret = args.client_secret
    if not client_id:
        logging.warning('No client ID supplied')
        client_id = ''
    if not client_secret:
        logging.warning('No client secret supplied')
        client_secret = ''
    client_info = util.ClientInfo.Create(discovery_doc, args.scope, client_id, client_secret, args.user_agent, names, args.api_key)
    outdir = os.path.expanduser(args.outdir) or client_info.default_directory
    if os.path.exists(outdir) and (not args.overwrite):
        raise exceptions.ConfigurationValueError('Output directory exists, pass --overwrite to replace the existing files.')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return gen_client_lib.DescriptorGenerator(discovery_doc, client_info, names, args.root_package, outdir, base_package=args.base_package, protorpc_package=args.protorpc_package, init_wildcards_file=args.init_file == 'wildcards', use_proto2=args.experimental_proto2_output, unelidable_request_methods=args.unelidable_request_methods, apitools_version=args.apitools_version)