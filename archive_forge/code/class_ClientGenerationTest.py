import importlib
import logging
import os
import six
import subprocess
import sys
import tempfile
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
class ClientGenerationTest(unittest.TestCase):

    def setUp(self):
        super(ClientGenerationTest, self).setUp()
        self.gen_client_binary = 'gen_client'

    @test_utils.SkipOnWindows
    def testGeneration(self):
        for api in _API_LIST:
            with test_utils.TempDir(change_to=True):
                args = [self.gen_client_binary, '--client_id=12345', '--client_secret=67890', '--discovery_url=%s' % api, '--outdir=generated', '--overwrite', 'client']
                logging.info('Testing API %s with command line: %s', api, ' '.join(args))
                retcode = gen_client.main(args)
                if retcode == 128:
                    logging.error('Failed to fetch discovery doc, continuing.')
                    continue
                self.assertEqual(0, retcode)
                sys.path.insert(0, os.path.join(os.getcwd(), 'generated'))
                importlib.import_module('{}_{}_client'.format(*api.split('.')))