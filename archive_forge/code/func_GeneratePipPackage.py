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
def GeneratePipPackage(args):
    """Generate a client as a pip-installable tarball."""
    discovery_doc = _GetDiscoveryDocFromFlags(args)
    package = discovery_doc['name']
    original_outdir = os.path.expanduser(args.outdir)
    args.outdir = os.path.join(args.outdir, 'apitools/clients/%s' % package)
    args.root_package = 'apitools.clients.%s' % package
    codegen = _GetCodegenFromFlags(args)
    if codegen is None:
        logging.error('Failed to create codegen, exiting.')
        return 1
    _WriteGeneratedFiles(args, codegen)
    _WriteInit(codegen)
    with util.Chdir(original_outdir):
        _WriteSetupPy(codegen)
        with util.Chdir('apitools'):
            _WriteIntermediateInit(codegen)
            with util.Chdir('clients'):
                _WriteIntermediateInit(codegen)