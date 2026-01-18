from __future__ import annotations
from babel.messages import frontend
class init_catalog(frontend.InitCatalog, Command):
    """New catalog initialization command for use in ``setup.py`` scripts.

    If correctly installed, this command is available to Setuptools-using
    setup scripts automatically. For projects using plain old ``distutils``,
    the command needs to be registered explicitly in ``setup.py``::

        from babel.messages.setuptools_frontend import init_catalog

        setup(
            ...
            cmdclass = {'init_catalog': init_catalog}
        )
    """