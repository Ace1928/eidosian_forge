import copy
import importlib
from kivy.logger import Logger
from kivy.context import register_context
import kivy.factory_registers  # NOQA
class FactoryBase(object):

    def __init__(self):
        super(FactoryBase, self).__init__()
        self.classes = {}

    @classmethod
    def create_from(cls, factory):
        """Creates a instance of the class, and initializes to the state of
        ``factory``.

        :param factory: The factory to initialize from.
        :return: A new instance of this class.
        """
        obj = cls()
        obj.classes = copy.copy(factory.classes)
        return obj

    def is_template(self, classname):
        """Return True if the classname is a template from the
        :class:`~kivy.lang.Builder`.

        .. versionadded:: 1.0.5
        """
        if classname in self.classes:
            return self.classes[classname]['is_template']
        else:
            return False

    def register(self, classname, cls=None, module=None, is_template=False, baseclasses=None, filename=None, warn=False):
        """Register a new classname referring to a real class or
        class definition in a module. Warn, if True will emit a warning message
        when a class is re-declared.

        .. versionchanged:: 1.9.0
            `warn` was added.

        .. versionchanged:: 1.7.0
            :attr:`baseclasses` and :attr:`filename` added

        .. versionchanged:: 1.0.5
            :attr:`is_template` has been added in 1.0.5.
        """
        if cls is None and module is None and (baseclasses is None):
            raise ValueError('You must specify either cls= or module= or baseclasses =')
        if classname in self.classes:
            if warn:
                info = self.classes[classname]
                Logger.warning('Factory: Ignored class "{}" re-declaration. Current -  module: {}, cls: {}, baseclass: {}, filename: {}. Ignored -  module: {}, cls: {}, baseclass: {}, filename: {}.'.format(classname, info['module'], info['cls'], info['baseclasses'], info['filename'], module, cls, baseclasses, filename))
            return
        self.classes[classname] = {'module': module, 'cls': cls, 'is_template': is_template, 'baseclasses': baseclasses, 'filename': filename}

    def unregister(self, *classnames):
        """Unregisters the classnames previously registered via the
        register method. This allows the same classnames to be re-used in
        different contexts.

        .. versionadded:: 1.7.1
        """
        for classname in classnames:
            if classname in self.classes:
                self.classes.pop(classname)

    def unregister_from_filename(self, filename):
        """Unregister all the factory objects related to the filename passed in
        the parameter.

        .. versionadded:: 1.7.0
        """
        to_remove = [x for x in self.classes if self.classes[x]['filename'] == filename]
        for name in to_remove:
            del self.classes[name]

    def __getattr__(self, name):
        classes = self.classes
        if name not in classes:
            if name[0] == name[0].lower():
                raise AttributeError('First letter of class name <%s> is in lowercase' % name)
            raise FactoryException('Unknown class <%s>' % name)
        item = classes[name]
        cls = item['cls']
        if cls is None:
            if item['module']:
                module = importlib.__import__(name=item['module'], fromlist='*', level=0)
                if not hasattr(module, name):
                    raise FactoryException('No class named <%s> in module <%s>' % (name, item['module']))
                cls = item['cls'] = getattr(module, name)
            elif item['baseclasses']:
                rootwidgets = []
                for basecls in item['baseclasses'].split('+'):
                    rootwidgets.append(Factory.get(basecls))
                cls = item['cls'] = type(str(name), tuple(rootwidgets), {})
            else:
                raise FactoryException('No information to create the class')
        return cls
    get = __getattr__