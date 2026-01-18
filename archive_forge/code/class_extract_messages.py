from __future__ import annotations
from babel.messages import frontend
class extract_messages(frontend.ExtractMessages, Command):
    """Message extraction command for use in ``setup.py`` scripts.

    If correctly installed, this command is available to Setuptools-using
    setup scripts automatically. For projects using plain old ``distutils``,
    the command needs to be registered explicitly in ``setup.py``::

        from babel.messages.setuptools_frontend import extract_messages

        setup(
            ...
            cmdclass = {'extract_messages': extract_messages}
        )
    """