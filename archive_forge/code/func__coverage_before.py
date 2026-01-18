from distutils import cmd
import distutils.errors
import logging
import os
import sys
import warnings
def _coverage_before(self):
    logger.debug('_coverage_before called')
    package = self.distribution.get_name()
    if package.startswith('python-'):
        package = package[7:]
    if self.coverage_package_name:
        package = self.coverage_package_name
    options = '--source %s --parallel-mode' % package
    os.environ['PYTHON'] = 'coverage run %s' % options
    logger.debug("os.environ['PYTHON'] = %r", os.environ['PYTHON'])