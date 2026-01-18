import os
import sys
import warnings
from pathlib import Path
from threading import local
from IPython.core import page, payloadpage
from IPython.core.autocall import ZMQExitAutocall
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magics import CodeMagics, MacroToEdit  # type:ignore[attr-defined]
from IPython.core.usage import default_banner
from IPython.display import Javascript, display
from IPython.utils import openpy
from IPython.utils.process import arg_split, system  # type:ignore[attr-defined]
from jupyter_client.session import Session, extract_header
from jupyter_core.paths import jupyter_runtime_dir
from traitlets import Any, CBool, CBytes, Dict, Instance, Type, default, observe
from ipykernel import connect_qtconsole, get_connection_file, get_connection_info
from ipykernel.displayhook import ZMQShellDisplayHook
from ipykernel.jsonutil import encode_images, json_clean
class ZMQDisplayPublisher(DisplayPublisher):
    """A display publisher that publishes data using a ZeroMQ PUB socket."""
    session = Instance(Session, allow_none=True)
    pub_socket = Any(allow_none=True)
    parent_header = Dict({})
    topic = CBytes(b'display_data')
    _thread_local = Any()

    def set_parent(self, parent):
        """Set the parent for outbound messages."""
        self.parent_header = extract_header(parent)

    def _flush_streams(self):
        """flush IO Streams prior to display"""
        sys.stdout.flush()
        sys.stderr.flush()

    @default('_thread_local')
    def _default_thread_local(self):
        """Initialize our thread local storage"""
        return local()

    @property
    def _hooks(self):
        if not hasattr(self._thread_local, 'hooks'):
            self._thread_local.hooks = []
        return self._thread_local.hooks

    def publish(self, data, metadata=None, transient=None, update=False):
        """Publish a display-data message

        Parameters
        ----------
        data : dict
            A mime-bundle dict, keyed by mime-type.
        metadata : dict, optional
            Metadata associated with the data.
        transient : dict, optional, keyword-only
            Transient data that may only be relevant during a live display,
            such as display_id.
            Transient data should not be persisted to documents.
        update : bool, optional, keyword-only
            If True, send an update_display_data message instead of display_data.
        """
        self._flush_streams()
        if metadata is None:
            metadata = {}
        if transient is None:
            transient = {}
        self._validate_data(data, metadata)
        content = {}
        content['data'] = encode_images(data)
        content['metadata'] = metadata
        content['transient'] = transient
        msg_type = 'update_display_data' if update else 'display_data'
        assert self.session is not None
        msg = self.session.msg(msg_type, json_clean(content), parent=self.parent_header)
        for hook in self._hooks:
            msg = hook(msg)
            if msg is None:
                return
        self.session.send(self.pub_socket, msg, ident=self.topic)

    def clear_output(self, wait=False):
        """Clear output associated with the current execution (cell).

        Parameters
        ----------
        wait : bool (default: False)
            If True, the output will not be cleared immediately,
            instead waiting for the next display before clearing.
            This reduces bounce during repeated clear & display loops.

        """
        content = dict(wait=wait)
        self._flush_streams()
        assert self.session is not None
        msg = self.session.msg('clear_output', json_clean(content), parent=self.parent_header)
        for hook in self._hooks:
            msg = hook(msg)
            if msg is None:
                return
        self.session.send(self.pub_socket, msg, ident=self.topic)

    def register_hook(self, hook):
        """
        Registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object

        Returns
        -------
        Either a publishable message, or `None`.
        The DisplayHook objects must return a message from
        the __call__ method if they still require the
        `session.send` method to be called after transformation.
        Returning `None` will halt that execution path, and
        session.send will not be called.
        """
        self._hooks.append(hook)

    def unregister_hook(self, hook):
        """
        Un-registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object which has previously been
            registered as a hook.

        Returns
        -------
        bool - `True` if the hook was removed, `False` if it wasn't
            found.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False