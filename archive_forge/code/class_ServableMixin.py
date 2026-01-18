from __future__ import annotations
import asyncio
import functools
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import (
import param  # type: ignore
from bokeh.core.serialization import DeserializationError
from bokeh.document import Document
from bokeh.resources import Resources
from jinja2 import Template
from pyviz_comms import Comm  # type: ignore
from ._param import Align, Aspect, Margin
from .config import config, panel_extension
from .io import serve
from .io.document import create_doc_if_none_exists, init_doc
from .io.embed import embed_state
from .io.loading import start_loading_spinner, stop_loading_spinner
from .io.model import add_to_doc, patch_cds_msg
from .io.notebook import (
from .io.save import save
from .io.state import curdoc_locked, set_curdoc, state
from .util import escape, param_reprs
from .util.parameters import get_params_to_inherit
class ServableMixin:
    """
    Mixin to define methods shared by objects which can served.
    """

    def _modify_doc(self, server_id: str, title: str, doc: Document, location: Optional['Location']) -> Document:
        """
        Callback to handle FunctionHandler document creation.
        """
        if server_id:
            state._servers[server_id][2].append(doc)
        return self.server_doc(doc, title, location)

    def _add_location(self, doc: Document, location: Optional['Location' | bool], root: Optional['Model']=None) -> 'Location':
        from .io.location import Location
        if isinstance(location, Location):
            loc = location
            state._locations[doc] = loc
        elif doc in state._locations:
            loc = state._locations[doc]
        else:
            with set_curdoc(doc):
                loc = state.location
        if root is None:
            loc_model = loc.get_root(doc)
        else:
            loc_model = loc._get_model(doc, root)
        loc_model.name = 'location'
        doc.on_session_destroyed(loc._server_destroy)
        doc.add_root(loc_model)
        return loc

    def servable(self, title: Optional[str]=None, location: bool | 'Location'=True, area: str='main', target: Optional[str]=None) -> 'ServableMixin':
        """
        Serves the object or adds it to the configured
        pn.state.template if in a `panel serve` context, writes to the
        DOM if in a pyodide context and returns the Panel object to
        allow it to display itself in a notebook context.

        Arguments
        ---------
        title : str
          A string title to give the Document (if served as an app)
        location : boolean or panel.io.location.Location
          Whether to create a Location component to observe and
          set the URL location.
        area: str (deprecated)
          The area of a template to add the component too. Only has an
          effect if pn.config.template has been set.
        target: str
          Target area to write to. If a template has been configured
          on pn.config.template this refers to the target area in the
          template while in pyodide this refers to the ID of the DOM
          node to write to.

        Returns
        -------
        The Panel object itself
        """
        if curdoc_locked().session_context:
            logger = logging.getLogger('bokeh')
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.WARN)
            if config.template:
                area = target or area or 'main'
                template = state.template
                assert template is not None
                if template.title == template.param.title.default and title:
                    template.title = title
                if area == 'main':
                    template.main.append(self)
                elif area == 'sidebar':
                    template.sidebar.append(self)
                elif area == 'modal':
                    template.modal.append(self)
                elif area == 'header':
                    template.header.append(self)
            else:
                self.server_doc(title=title, location=location)
        elif state._is_pyodide and 'pyodide_kernel' not in sys.modules:
            from .io.pyodide import _IN_PYSCRIPT_WORKER, _IN_WORKER, _get_pyscript_target, write
            if _IN_WORKER and (not _IN_PYSCRIPT_WORKER):
                return self
            try:
                target = target or _get_pyscript_target()
            except Exception:
                target = None
            if target is not None:
                task = asyncio.create_task(write(target, self))
                _tasks.add(task)
                task.add_done_callback(_tasks.discard)
        return self

    def show(self, title: Optional[str]=None, port: int=0, address: Optional[str]=None, websocket_origin: Optional[str]=None, threaded: bool=False, verbose: bool=True, open: bool=True, location: bool | 'Location'=True, **kwargs) -> 'StoppableThread' | 'Server':
        """
        Starts a Bokeh server and displays the Viewable in a new tab.

        Arguments
        ---------
        title : str | None
          A string title to give the Document (if served as an app)
        port: int (optional, default=0)
          Allows specifying a specific port
        address : str
          The address the server should listen on for HTTP requests.
        websocket_origin: str or list(str) (optional)
          A list of hosts that can connect to the websocket.
          This is typically required when embedding a server app in
          an external web site.
          If None, "localhost" is used.
        threaded: boolean (optional, default=False)
          Whether to launch the Server on a separate thread, allowing
          interactive use.
        verbose: boolean (optional, default=True)
          Whether to print the address and port
        open : boolean (optional, default=True)
          Whether to open the server in a new browser tab
        location : boolean or panel.io.location.Location
          Whether to create a Location component to observe and
          set the URL location.

        Returns
        -------
        server: bokeh.server.Server or panel.io.server.StoppableThread
          Returns the Bokeh server instance or the thread the server
          was launched on (if threaded=True)
        """
        return serve(self, port=port, address=address, websocket_origin=websocket_origin, show=open, start=True, title=title, verbose=verbose, location=location, threaded=threaded, **kwargs)