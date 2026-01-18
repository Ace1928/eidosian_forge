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
def _GetDiscoveryDocFromFlags(args):
    """Get the discovery doc from flags."""
    if args.discovery_url:
        try:
            return util.FetchDiscoveryDoc(args.discovery_url)
        except exceptions.CommunicationError:
            raise exceptions.GeneratedClientError('Could not fetch discovery doc')
    infile = os.path.expanduser(args.infile) or '/dev/stdin'
    with io.open(infile, encoding='utf8') as f:
        return json.loads(util.ReplaceHomoglyphs(f.read()))