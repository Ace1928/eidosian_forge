import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
class DockArea(Container, QtWidgets.QWidget):

    def __init__(self, parent=None, temporary=False, home=None):
        Container.__init__(self, self)
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.dockdrop = DockDrop(self)
        self.dockdrop.removeAllowedArea('center')
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.docks = weakref.WeakValueDictionary()
        self.topContainer = None
        self.dockdrop.raiseOverlay()
        self.temporary = temporary
        self.tempAreas = []
        self.home = home

    def type(self):
        return 'top'

    def addDock(self, dock=None, position='bottom', relativeTo=None, **kwds):
        """Adds a dock to this area.
        
        ============== =================================================================
        **Arguments:**
        dock           The new Dock object to add. If None, then a new Dock will be 
                       created.
        position       'bottom', 'top', 'left', 'right', 'above', or 'below'
        relativeTo     If relativeTo is None, then the new Dock is added to fill an 
                       entire edge of the window. If relativeTo is another Dock, then 
                       the new Dock is placed adjacent to it (or in a tabbed 
                       configuration for 'above' and 'below'). 
        ============== =================================================================
        
        All extra keyword arguments are passed to Dock.__init__() if *dock* is
        None.        
        """
        if dock is None:
            dock = Dock(**kwds)
        if not self.temporary:
            dock.orig_area = self
        if relativeTo is None or relativeTo is self:
            if self.topContainer is None:
                container = self
                neighbor = None
            else:
                container = self.topContainer
                neighbor = None
        else:
            if isinstance(relativeTo, str):
                relativeTo = self.docks[relativeTo]
            container = self.getContainer(relativeTo)
            if container is None:
                raise TypeError('Dock %s is not contained in a DockArea; cannot add another dock relative to it.' % relativeTo)
            neighbor = relativeTo
        neededContainer = {'bottom': 'vertical', 'top': 'vertical', 'left': 'horizontal', 'right': 'horizontal', 'above': 'tab', 'below': 'tab'}[position]
        if neededContainer != container.type() and container.type() == 'tab':
            neighbor = container
            container = container.container()
        if neededContainer != container.type():
            if neighbor is None:
                container = self.addContainer(neededContainer, self.topContainer)
            else:
                container = self.addContainer(neededContainer, neighbor)
        insertPos = {'bottom': 'after', 'top': 'before', 'left': 'before', 'right': 'after', 'above': 'before', 'below': 'after'}[position]
        old = dock.container()
        container.insert(dock, insertPos, neighbor)
        self.docks[dock.name()] = dock
        if old is not None:
            old.apoptose()
        return dock

    def moveDock(self, dock, position, neighbor):
        """
        Move an existing Dock to a new location. 
        """
        if position in ['left', 'right', 'top', 'bottom'] and neighbor is not None and (neighbor.container() is not None) and (neighbor.container().type() == 'tab'):
            neighbor = neighbor.container()
        self.addDock(dock, position, neighbor)

    def getContainer(self, obj):
        if obj is None:
            return self
        return obj.container()

    def makeContainer(self, typ):
        if typ == 'vertical':
            new = VContainer(self)
        elif typ == 'horizontal':
            new = HContainer(self)
        elif typ == 'tab':
            new = TContainer(self)
        else:
            raise ValueError("typ must be one of 'vertical', 'horizontal', or 'tab'")
        return new

    def addContainer(self, typ, obj):
        """Add a new container around obj"""
        new = self.makeContainer(typ)
        container = self.getContainer(obj)
        container.insert(new, 'before', obj)
        if obj is not None:
            new.insert(obj)
        self.dockdrop.raiseOverlay()
        return new

    def insert(self, new, pos=None, neighbor=None):
        if self.topContainer is not None:
            self.topContainer.containerChanged(None)
        self.layout.addWidget(new)
        new.containerChanged(self)
        self.topContainer = new
        self.dockdrop.raiseOverlay()

    def count(self):
        if self.topContainer is None:
            return 0
        return 1

    def resizeEvent(self, ev):
        self.dockdrop.resizeOverlay(self.size())

    def addTempArea(self):
        if self.home is None:
            area = DockArea(temporary=True, home=self)
            self.tempAreas.append(area)
            win = TempAreaWindow(area)
            area.win = win
            win.show()
        else:
            area = self.home.addTempArea()
        return area

    def floatDock(self, dock):
        """Removes *dock* from this DockArea and places it in a new window."""
        area = self.addTempArea()
        area.win.resize(dock.size())
        area.moveDock(dock, 'top', None)

    def removeTempArea(self, area):
        self.tempAreas.remove(area)
        area.window().close()

    def saveState(self):
        """
        Return a serialized (storable) representation of the state of
        all Docks in this DockArea."""
        if self.topContainer is None:
            main = None
        else:
            main = self.childState(self.topContainer)
        state = {'main': main, 'float': []}
        for a in self.tempAreas:
            geo = a.win.geometry()
            geo = (geo.x(), geo.y(), geo.width(), geo.height())
            state['float'].append((a.saveState(), geo))
        return state

    def childState(self, obj):
        if isinstance(obj, Dock):
            return ('dock', obj.name(), {})
        else:
            childs = []
            for i in range(obj.count()):
                childs.append(self.childState(obj.widget(i)))
            return (obj.type(), childs, obj.saveState())

    def restoreState(self, state, missing='error', extra='bottom'):
        """
        Restore Dock configuration as generated by saveState.
        
        This function does not create any Docks--it will only 
        restore the arrangement of an existing set of Docks.
        
        By default, docks that are described in *state* but do not exist
        in the dock area will cause an exception to be raised. This behavior
        can be changed by setting *missing* to 'ignore' or 'create'.
        
        Extra docks that are in the dockarea but that are not mentioned in
        *state* will be added to the bottom of the dockarea, unless otherwise
        specified by the *extra* argument.
        """
        containers, docks = self.findAll()
        oldTemps = self.tempAreas[:]
        if state['main'] is not None:
            self.buildFromState(state['main'], docks, self, missing=missing)
        for s in state['float']:
            a = self.addTempArea()
            a.buildFromState(s[0]['main'], docks, a, missing=missing)
            a.win.setGeometry(*s[1])
            a.apoptose()
        for d in docks.values():
            if extra == 'float':
                a = self.addTempArea()
                a.addDock(d, 'below')
            else:
                self.moveDock(d, extra, None)
        for c in containers:
            c.close()
        for a in oldTemps:
            a.apoptose()

    def buildFromState(self, state, docks, root, depth=0, missing='error'):
        typ, contents, state = state
        if typ == 'dock':
            try:
                obj = docks[contents]
                del docks[contents]
            except KeyError:
                if missing == 'error':
                    raise Exception('Cannot restore dock state; no dock with name "%s"' % contents)
                elif missing == 'create':
                    obj = Dock(name=contents)
                elif missing == 'ignore':
                    return
                else:
                    raise ValueError('"missing" argument must be one of "error", "create", or "ignore".')
        else:
            obj = self.makeContainer(typ)
        root.insert(obj, 'after')
        if typ != 'dock':
            for o in contents:
                self.buildFromState(o, docks, obj, depth + 1, missing=missing)
            obj.apoptose(propagate=False)
            obj.restoreState(state)

    def findAll(self, obj=None, c=None, d=None):
        if obj is None:
            obj = self.topContainer
        if c is None:
            c = []
            d = {}
            for a in self.tempAreas:
                c1, d1 = a.findAll()
                c.extend(c1)
                d.update(d1)
        if isinstance(obj, Dock):
            d[obj.name()] = obj
        elif obj is not None:
            c.append(obj)
            for i in range(obj.count()):
                o2 = obj.widget(i)
                c2, d2 = self.findAll(o2)
                c.extend(c2)
                d.update(d2)
        return (c, d)

    def apoptose(self, propagate=True):
        if self.topContainer is None or self.topContainer.count() == 0:
            self.topContainer = None
            if self.temporary and self.home is not None:
                self.home.removeTempArea(self)

    def clear(self):
        docks = self.findAll()[1]
        for dock in docks.values():
            dock.close()

    def dragEnterEvent(self, *args):
        self.dockdrop.dragEnterEvent(*args)

    def dragMoveEvent(self, *args):
        self.dockdrop.dragMoveEvent(*args)

    def dragLeaveEvent(self, *args):
        self.dockdrop.dragLeaveEvent(*args)

    def dropEvent(self, *args):
        self.dockdrop.dropEvent(*args)

    def printState(self, state=None, name='Main'):
        if state is None:
            state = self.saveState()
        print('=== %s dock area ===' % name)
        if state['main'] is None:
            print('   (empty)')
        else:
            self._printAreaState(state['main'])
        for i, float in enumerate(state['float']):
            self.printState(float[0], name='float %d' % i)

    def _printAreaState(self, area, indent=0):
        if area[0] == 'dock':
            print('  ' * indent + area[0] + ' ' + str(area[1:]))
            return
        else:
            print('  ' * indent + area[0])
            for ch in area[1]:
                self._printAreaState(ch, indent + 1)