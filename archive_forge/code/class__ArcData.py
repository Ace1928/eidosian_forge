from pyomo.network.port import Port
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from weakref import ref as weakref_ref
import logging, sys
class _ArcData(ActiveComponentData):
    """
    This class defines the data for a single Arc

    Attributes
    ----------
        source: `Port`
            The source Port when directed, else None. Aliases to src.
        destination: `Port`
            The destination Port when directed, else None. Aliases to dest.
        ports: `tuple`
            A tuple containing both ports. If directed, this is in the
            order (source, destination).
        directed: `bool`
            True if directed, False if not
        expanded_block: `Block`
            A reference to the block on which expanded constraints for this
            arc were placed
    """
    __slots__ = ('_ports', '_directed', '_expanded_block')

    def __init__(self, component=None, **kwds):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._active = True
        self._ports = None
        self._directed = None
        self._expanded_block = None
        if len(kwds):
            self.set_value(kwds)

    def __getattr__(self, name):
        """Returns `self.expanded_block.name` if it exists"""
        eb = self.expanded_block
        if eb is not None:
            try:
                return getattr(eb, name)
            except AttributeError:
                pass
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    @property
    def source(self):
        return self._ports[0] if self._directed and self._ports is not None else None
    src = source

    @property
    def destination(self):
        return self._ports[1] if self._directed and self._ports is not None else None
    dest = destination

    @property
    def ports(self):
        return self._ports

    @property
    def directed(self):
        return self._directed

    @property
    def expanded_block(self):
        return self._expanded_block

    def set_value(self, vals):
        """Set the port attributes on this arc"""
        d = self._directed if self._directed is not None else self.parent_component()._init_directed
        vals = _iterable_to_dict(vals, d, self.name)
        source = vals.pop('source', None)
        destination = vals.pop('destination', None)
        ports = vals.pop('ports', None)
        directed = vals.pop('directed', None)
        if len(vals):
            raise ValueError('set_value passed unrecognized keywords in val:\n\t' + '\n\t'.join(('%s = %s' % (k, v) for k, v in vals.items())))
        if directed is not None:
            if source is None and destination is None:
                if directed and ports is not None:
                    try:
                        source, destination = ports
                        ports = None
                    except:
                        raise ValueError("Failed to unpack 'ports' argument of arc '%s'. Argument must be a 2-member tuple or list." % self.name)
            elif not directed:
                raise ValueError("Passed False value for 'directed' for arc '%s', but specified source or destination." % self.name)
        self._validate_ports(source, destination, ports)
        if self.ports is not None:
            weakref_self = weakref_ref(self)
            for port in self.ports:
                port._arcs.remove(weakref_self)
            if self._directed:
                self.source._dests.remove(weakref_self)
                self.destination._sources.remove(weakref_self)
        self._ports = tuple(ports) if ports is not None else (source, destination)
        self._directed = source is not None
        weakref_self = weakref_ref(self)
        for port in self._ports:
            port._arcs.append(weakref_self)
        if self._directed:
            source._dests.append(weakref_self)
            destination._sources.append(weakref_self)

    def _validate_ports(self, source, destination, ports):
        msg = 'Arc %s: ' % self.name
        if ports is not None:
            if source is not None or destination is not None:
                raise ValueError(msg + "cannot specify 'source' or 'destination' when using 'ports' argument.")
            if type(ports) not in (list, tuple) or len(ports) != 2:
                raise ValueError(msg + "argument 'ports' must be list or tuple containing exactly 2 Ports.")
            for p in ports:
                try:
                    if p.ctype is not Port:
                        raise ValueError(msg + "found object '%s' in 'ports' not of type Port." % p.name)
                    elif p.is_indexed():
                        raise ValueError(msg + "found indexed Port '%s' in 'ports', must use single Ports for Arc." % p.name)
                except AttributeError:
                    raise ValueError(msg + "found object '%s' in 'ports' not of type Port." % str(p))
        else:
            if source is None or destination is None:
                raise ValueError(msg + "must specify both 'source' and 'destination' for directed Arc.")
            for p, side in [(source, 'source'), (destination, 'destination')]:
                try:
                    if p.ctype is not Port:
                        raise ValueError(msg + "%s object '%s' not of type Port." % (p.name, side))
                    elif p.is_indexed():
                        raise ValueError(msg + "found indexed Port '%s' as %s, must use single Ports for Arc." % (source.name, side))
                except AttributeError:
                    raise ValueError(msg + "%s object '%s' not of type Port." % (str(p), side))