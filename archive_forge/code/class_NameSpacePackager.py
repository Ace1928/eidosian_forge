from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
class NameSpacePackager(object):

    def __init__(self, pkg_data):
        assert isinstance(pkg_data, dict)
        self._pkg_data = pkg_data
        self.full_package_name = self.pn(self._pkg_data['full_package_name'])
        self._split = None
        self.depth = self.full_package_name.count('.')
        self.nested = self._pkg_data.get('nested', False)
        self.command = None
        self.python_version()
        self._pkg = [None, None]
        if sys.argv[0] == 'setup.py' and sys.argv[1] == 'install' and ('--single-version-externally-managed' not in sys.argv):
            if os.environ.get('READTHEDOCS', None) == 'True':
                os.system('pip install .')
                sys.exit(0)
            if not os.environ.get('RUAMEL_NO_PIP_INSTALL_CHECK', False):
                print('error: you have to install with "pip install ."')
                sys.exit(1)
        if self._pkg_data.get('universal'):
            Distribution.is_pure = lambda *args: True
        else:
            Distribution.is_pure = lambda *args: False
        for x in sys.argv:
            if x[0] == '-' or x == 'setup.py':
                continue
            self.command = x
            break

    def pn(self, s):
        if sys.version_info < (3,) and isinstance(s, unicode):
            return s.encode('utf-8')
        return s

    @property
    def split(self):
        """split the full package name in list of compontents traditionally
        done by setuptools.find_packages. This routine skips any directories
        with __init__.py, for which the name starts with "_" or ".", or contain a
        setup.py/tox.ini (indicating a subpackage)
        """
        skip = []
        if self._split is None:
            fpn = self.full_package_name.split('.')
            self._split = []
            while fpn:
                self._split.insert(0, '.'.join(fpn))
                fpn = fpn[:-1]
            for d in sorted(os.listdir('.')):
                if not os.path.isdir(d) or d == self._split[0] or d[0] in '._':
                    continue
                x = os.path.join(d, '__init__.py')
                if os.path.exists(x):
                    pd = _package_data(x)
                    if pd.get('nested', False):
                        skip.append(d)
                        continue
                    self._split.append(self.full_package_name + '.' + d)
            if sys.version_info < (3,):
                self._split = [y.encode('utf-8') if isinstance(y, unicode) else y for y in self._split]
        if skip:
            pass
        return self._split

    @property
    def namespace_packages(self):
        return self.split[:self.depth]

    def namespace_directories(self, depth=None):
        """return list of directories where the namespace should be created /
        can be found
        """
        res = []
        for index, d in enumerate(self.split[:depth]):
            if index > 0:
                d = os.path.join(*d.split('.'))
            res.append('.' + d)
        return res

    @property
    def package_dir(self):
        d = {self.full_package_name: '.'}
        if 'extra_packages' in self._pkg_data:
            return d
        if len(self.split) > 1:
            d[self.split[0]] = self.namespace_directories(1)[0]
        return d

    def create_dirs(self):
        """create the directories necessary for namespace packaging"""
        directories = self.namespace_directories(self.depth)
        if not directories:
            return
        if not os.path.exists(directories[0]):
            for d in directories:
                os.mkdir(d)
                with open(os.path.join(d, '__init__.py'), 'w') as fp:
                    fp.write('import pkg_resources\npkg_resources.declare_namespace(__name__)\n')

    def python_version(self):
        supported = self._pkg_data.get('supported')
        if supported is None:
            return
        if len(supported) == 1:
            minimum = supported[0]
        else:
            for x in supported:
                if x[0] == sys.version_info[0]:
                    minimum = x
                    break
            else:
                return
        if sys.version_info < minimum:
            print('minimum python version(s): ' + str(supported))
            sys.exit(1)

    def check(self):
        try:
            from pip.exceptions import InstallationError
        except ImportError:
            return
        if self.command not in ['install', 'develop']:
            return
        prefix = self.split[0]
        prefixes = set([prefix, prefix.replace('_', '-')])
        for p in sys.path:
            if not p:
                continue
            if os.path.exists(os.path.join(p, 'setup.py')):
                continue
            if not os.path.isdir(p):
                continue
            if p.startswith('/tmp/'):
                continue
            for fn in os.listdir(p):
                for pre in prefixes:
                    if fn.startswith(pre):
                        break
                else:
                    continue
                full_name = os.path.join(p, fn)
                if fn == prefix and os.path.isdir(full_name):
                    if self.command == 'develop':
                        raise InstallationError('Cannot mix develop (pip install -e),\nwith non-develop installs for package name {0}'.format(fn))
                elif fn == prefix:
                    raise InstallationError('non directory package {0} in {1}'.format(fn, p))
                for pre in [x + '.' for x in prefixes]:
                    if fn.startswith(pre):
                        break
                else:
                    continue
                if fn.endswith('-link') and self.command == 'install':
                    raise InstallationError('Cannot mix non-develop with develop\n(pip install -e) installs for package name {0}'.format(fn))

    def entry_points(self, script_name=None, package_name=None):
        """normally called without explicit script_name and package name
        the default console_scripts entry depends on the existence of __main__.py:
        if that file exists then the function main() in there is used, otherwise
        the in __init__.py.

        the _package_data entry_points key/value pair can be explicitly specified
        including a "=" character. If the entry is True or 1 the
        scriptname is the last part of the full package path (split on '.')
        if the ep entry is a simple string without "=", that is assumed to be
        the name of the script.
        """

        def pckg_entry_point(name):
            return '{0}{1}:main'.format(name, '.__main__' if os.path.exists('__main__.py') else '')
        ep = self._pkg_data.get('entry_points', True)
        if isinstance(ep, dict):
            return ep
        if ep is None:
            return None
        if ep not in [True, 1]:
            if '=' in ep:
                return {'console_scripts': [ep]}
            script_name = ep
        if package_name is None:
            package_name = self.full_package_name
        if not script_name:
            script_name = package_name.split('.')[-1]
        return {'console_scripts': ['{0} = {1}'.format(script_name, pckg_entry_point(package_name))]}

    @property
    def url(self):
        if self.full_package_name.startswith('ruamel.'):
            sp = self.full_package_name.split('.', 1)
        else:
            sp = ['ruamel', self.full_package_name]
        return 'https://bitbucket.org/{0}/{1}'.format(*sp)

    @property
    def author(self):
        return self._pkg_data['author']

    @property
    def author_email(self):
        return self._pkg_data['author_email']

    @property
    def license(self):
        """return the license field from _package_data, None means MIT"""
        lic = self._pkg_data.get('license')
        if lic is None:
            return 'MIT license'
        return lic

    def has_mit_lic(self):
        return 'MIT' in self.license

    @property
    def description(self):
        return self._pkg_data['description']

    @property
    def status(self):
        status = self._pkg_data.get('status', 'β').lower()
        if status in ['α', 'alpha']:
            return (3, 'Alpha')
        elif status in ['β', 'beta']:
            return (4, 'Beta')
        elif 'stable' in status.lower():
            return (5, 'Production/Stable')
        raise NotImplementedError

    @property
    def classifiers(self):
        """this needs more intelligence, probably splitting the classifiers from _pkg_data
        and only adding defaults when no explicit entries were provided.
        Add explicit Python versions in sync with tox.env generation based on python_requires?
        """
        return sorted(set(['Development Status :: {0} - {1}'.format(*self.status), 'Intended Audience :: Developers', 'License :: ' + ('OSI Approved :: MIT' if self.has_mit_lic() else 'Other/Proprietary') + ' License', 'Operating System :: OS Independent', 'Programming Language :: Python'] + [self.pn(x) for x in self._pkg_data.get('classifiers', [])]))

    @property
    def keywords(self):
        return self.pn(self._pkg_data.get('keywords', []))

    @property
    def install_requires(self):
        """list of packages required for installation"""
        return self._analyse_packages[0]

    @property
    def install_pre(self):
        """list of packages required for installation"""
        return self._analyse_packages[1]

    @property
    def _analyse_packages(self):
        """gather from configuration, names starting with * need
        to be installed explicitly as they are not on PyPI
        install_requires should be  dict, with keys 'any', 'py27' etc
        or a list (which is as if only 'any' was defined

        ToDo: update with: pep508 conditional dependencies
        """
        if self._pkg[0] is None:
            self._pkg[0] = []
            self._pkg[1] = []
        ir = self._pkg_data.get('install_requires')
        if ir is None:
            return self._pkg
        if isinstance(ir, list):
            self._pkg[0] = ir
            return self._pkg
        packages = ir.get('any', [])
        if isinstance(packages, string_type):
            packages = packages.split()
        if self.nested:
            parent_pkg = self.full_package_name.rsplit('.', 1)[0]
            if parent_pkg not in packages:
                packages.append(parent_pkg)
        implementation = platform.python_implementation()
        if implementation == 'CPython':
            pyver = 'py{0}{1}'.format(*sys.version_info)
        elif implementation == 'PyPy':
            pyver = 'pypy' if sys.version_info < (3,) else 'pypy3'
        elif implementation == 'Jython':
            pyver = 'jython'
        packages.extend(ir.get(pyver, []))
        for p in packages:
            if p[0] == '*':
                p = p[1:]
                self._pkg[1].append(p)
            self._pkg[0].append(p)
        return self._pkg

    @property
    def extras_require(self):
        """dict of conditions -> extra packages informaton required for installation
        as of setuptools 33 doing `package ; python_version<=2.7' in install_requires
        still doesn't work

        https://www.python.org/dev/peps/pep-0508/
        https://wheel.readthedocs.io/en/latest/index.html#defining-conditional-dependencies
        https://hynek.me/articles/conditional-python-dependencies/
        """
        ep = self._pkg_data.get('extras_require')
        return ep

    @property
    def data_files(self):
        df = self._pkg_data.get('data_files', [])
        if self.has_mit_lic():
            df.append('LICENSE')
        if not df:
            return None
        return [('.', df)]

    @property
    def package_data(self):
        df = self._pkg_data.get('data_files', [])
        if self.has_mit_lic():
            df.append('LICENSE')
            exclude_files.append('LICENSE')
        pd = self._pkg_data.get('package_data', {})
        if df:
            pd[self.full_package_name] = df
        if sys.version_info < (3,):
            for k in pd:
                if isinstance(k, unicode):
                    pd[str(k)] = pd.pop(k)
        return pd

    @property
    def packages(self):
        s = self.split
        return s + self._pkg_data.get('extra_packages', [])

    @property
    def python_requires(self):
        return self._pkg_data.get('python_requires', None)

    @property
    def ext_modules(self):
        """
        Check if all modules specified in the value for 'ext_modules' can be build.
        That value (if not None) is a list of dicts with 'name', 'src', 'lib'
        Optional 'test' can be used to make sure trying to compile will work on the host

        creates and return the external modules as Extensions, unless that
        is not necessary at all for the action (like --version)

        test existence of compiler by using export CC=nonexistent; export CXX=nonexistent
        """
        if hasattr(self, '_ext_modules'):
            return self._ext_modules
        if '--version' in sys.argv:
            return None
        if platform.python_implementation() == 'Jython':
            return None
        try:
            plat = sys.argv.index('--plat-name')
            if 'win' in sys.argv[plat + 1]:
                return None
        except ValueError:
            pass
        self._ext_modules = []
        no_test_compile = False
        if '--restructuredtext' in sys.argv:
            no_test_compile = True
        elif 'sdist' in sys.argv:
            no_test_compile = True
        if no_test_compile:
            for target in self._pkg_data.get('ext_modules', []):
                ext = Extension(self.pn(target['name']), sources=[self.pn(x) for x in target['src']], libraries=[self.pn(x) for x in target.get('lib')])
                self._ext_modules.append(ext)
            return self._ext_modules
        print('sys.argv', sys.argv)
        import tempfile
        import shutil
        from textwrap import dedent
        import distutils.sysconfig
        import distutils.ccompiler
        from distutils.errors import CompileError, LinkError
        for target in self._pkg_data.get('ext_modules', []):
            ext = Extension(self.pn(target['name']), sources=[self.pn(x) for x in target['src']], libraries=[self.pn(x) for x in target.get('lib')])
            if 'test' not in target:
                self._ext_modules.append(ext)
                continue
            if sys.version_info[:2] == (3, 4) and platform.system() == 'Windows':
                if 'FORCE_C_BUILD_TEST' not in os.environ:
                    self._ext_modules.append(ext)
                    continue
            c_code = dedent(target['test'])
            try:
                tmp_dir = tempfile.mkdtemp(prefix='tmp_ruamel_')
                bin_file_name = 'test' + self.pn(target['name'])
                print('test compiling', bin_file_name)
                file_name = os.path.join(tmp_dir, bin_file_name + '.c')
                with open(file_name, 'w') as fp:
                    fp.write(c_code)
                compiler = distutils.ccompiler.new_compiler()
                assert isinstance(compiler, distutils.ccompiler.CCompiler)
                distutils.sysconfig.customize_compiler(compiler)
                compiler.add_include_dir(os.getcwd())
                if sys.version_info < (3,):
                    tmp_dir = tmp_dir.encode('utf-8')
                compile_out_dir = tmp_dir
                try:
                    compiler.link_executable(compiler.compile([file_name], output_dir=compile_out_dir), bin_file_name, output_dir=tmp_dir, libraries=ext.libraries)
                except CompileError:
                    debug('compile error:', file_name)
                    print('compile error:', file_name)
                    continue
                except LinkError:
                    debug('link error', file_name)
                    print('link error', file_name)
                    continue
                self._ext_modules.append(ext)
            except Exception as e:
                debug('Exception:', e)
                print('Exception:', e)
                if sys.version_info[:2] == (3, 4) and platform.system() == 'Windows':
                    traceback.print_exc()
            finally:
                shutil.rmtree(tmp_dir)
        return self._ext_modules

    @property
    def test_suite(self):
        return self._pkg_data.get('test_suite')

    def wheel(self, kw, setup):
        """temporary add setup.cfg if creating a wheel to include LICENSE file
        https://bitbucket.org/pypa/wheel/issues/47
        """
        if 'bdist_wheel' not in sys.argv:
            return False
        file_name = 'setup.cfg'
        if os.path.exists(file_name):
            return False
        with open(file_name, 'w') as fp:
            if os.path.exists('LICENSE'):
                fp.write('[metadata]\nlicense-file = LICENSE\n')
            else:
                print('\n\n>>>>>> LICENSE file not found <<<<<\n\n')
            if self._pkg_data.get('universal'):
                fp.write('[bdist_wheel]\nuniversal = 1\n')
        try:
            setup(**kw)
        except Exception:
            raise
        finally:
            os.remove(file_name)
        return True