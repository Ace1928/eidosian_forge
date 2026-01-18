import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
class PackageLoader(BaseLoader):
    """Load templates from a directory in a Python package.

    :param package_name: Import name of the package that contains the
        template directory.
    :param package_path: Directory within the imported package that
        contains the templates.
    :param encoding: Encoding of template files.

    The following example looks up templates in the ``pages`` directory
    within the ``project.ui`` package.

    .. code-block:: python

        loader = PackageLoader("project.ui", "pages")

    Only packages installed as directories (standard pip behavior) or
    zip/egg files (less common) are supported. The Python API for
    introspecting data in packages is too limited to support other
    installation methods the way this loader requires.

    There is limited support for :pep:`420` namespace packages. The
    template directory is assumed to only be in one namespace
    contributor. Zip files contributing to a namespace are not
    supported.

    .. versionchanged:: 3.0
        No longer uses ``setuptools`` as a dependency.

    .. versionchanged:: 3.0
        Limited PEP 420 namespace package support.
    """

    def __init__(self, package_name: str, package_path: 'str'='templates', encoding: str='utf-8') -> None:
        package_path = os.path.normpath(package_path).rstrip(os.path.sep)
        if package_path == os.path.curdir:
            package_path = ''
        elif package_path[:2] == os.path.curdir + os.path.sep:
            package_path = package_path[2:]
        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding
        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, 'An import spec was not found for the package.'
        loader = spec.loader
        assert loader is not None, 'A loader was not found for the package.'
        self._loader = loader
        self._archive = None
        template_root = None
        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))
            template_root = os.path.join(pkgdir, package_path).rstrip(os.path.sep)
        else:
            roots: t.List[str] = []
            if spec.submodule_search_locations:
                roots.extend(spec.submodule_search_locations)
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))
            for root in roots:
                root = os.path.join(root, package_path)
                if os.path.isdir(root):
                    template_root = root
                    break
        if template_root is None:
            raise ValueError(f'The {package_name!r} package was not installed in a way that PackageLoader understands.')
        self._template_root = template_root

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, str, t.Optional[t.Callable[[], bool]]]:
        p = os.path.normpath(posixpath.join(self._template_root, *split_template_path(template)))
        up_to_date: t.Optional[t.Callable[[], bool]]
        if self._archive is None:
            if not os.path.isfile(p):
                raise TemplateNotFound(template)
            with open(p, 'rb') as f:
                source = f.read()
            mtime = os.path.getmtime(p)

            def up_to_date() -> bool:
                return os.path.isfile(p) and os.path.getmtime(p) == mtime
        else:
            try:
                source = self._loader.get_data(p)
            except OSError as e:
                raise TemplateNotFound(template) from e
            up_to_date = None
        return (source.decode(self.encoding), p, up_to_date)

    def list_templates(self) -> t.List[str]:
        results: t.List[str] = []
        if self._archive is None:
            offset = len(self._template_root)
            for dirpath, _, filenames in os.walk(self._template_root):
                dirpath = dirpath[offset:].lstrip(os.path.sep)
                results.extend((os.path.join(dirpath, name).replace(os.path.sep, '/') for name in filenames))
        else:
            if not hasattr(self._loader, '_files'):
                raise TypeError('This zip import does not have the required metadata to list templates.')
            prefix = self._template_root[len(self._archive):].lstrip(os.path.sep) + os.path.sep
            offset = len(prefix)
            for name in self._loader._files.keys():
                if name.startswith(prefix) and name[-1] != os.path.sep:
                    results.append(name[offset:].replace(os.path.sep, '/'))
        results.sort()
        return results