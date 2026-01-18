import os
import re
import subprocess
import sys
from pbr.tests import base
def _check_wsgi_install_content(self, install_stdout):
    for cmd_name in self.cmd_names:
        install_txt = 'Installing %s script to %s' % (cmd_name, self.temp_dir)
        self.assertIn(install_txt, install_stdout)
        cmd_filename = os.path.join(self.temp_dir, 'bin', cmd_name)
        script_txt = open(cmd_filename, 'r').read()
        self.assertNotIn('pkg_resources', script_txt)
        main_block = 'if __name__ == "__main__":\n    import argparse\n    import socket\n    import sys\n    import wsgiref.simple_server as wss'
        if cmd_name == 'pbr_test_wsgi':
            app_name = 'main'
        else:
            app_name = 'WSGI.app'
        starting_block = 'STARTING test server pbr_testpackage.wsgi.%s' % app_name
        else_block = 'else:\n    application = None'
        self.assertIn(main_block, script_txt)
        self.assertIn(starting_block, script_txt)
        self.assertIn(else_block, script_txt)