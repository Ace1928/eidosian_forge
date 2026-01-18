from importlib import import_module
import inspect
import io
import token
import tokenize
import traceback
from sphinx.ext.autodoc import ClassLevelDocumenter
from sphinx.util import logging
from traits.has_traits import MetaHasTraits
from traits.trait_type import TraitType
from traits.traits import generic_trait
class TraitDocumenter(ClassLevelDocumenter):
    """ Specialized Documenter subclass for trait attributes.

    The class defines a new documenter that recovers the trait definition
    signature of module level and class level traits.

    To use the documenter, append the module path in the extension
    attribute of the `conf.py`.
    """
    objtype = 'traitattribute'
    directivetype = 'attribute'
    member_order = 60
    priority = 12

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """ Check that the documented member is a trait instance.
        """
        check = isattr and issubclass(type(member), TraitType) or _is_class_trait(membername, parent.object)
        return check

    def document_members(self, all_members=False):
        """ Trait attributes have no members """
        pass

    def import_object(self):
        """ Get the Trait object.

        Notes
        -----
        Code adapted from autodoc.Documenter.import_object.

        """
        try:
            current = self.module = import_module(self.modname)
            for part in self.objpath[:-1]:
                current = self.get_attr(current, part)
            name = self.objpath[-1]
            self.object_name = name
            self.object = None
            self.parent = current
            return True
        except Exception as err:
            if self.env.app and (not self.env.app.quiet):
                self.env.app.info(traceback.format_exc().rstrip())
            msg = 'autodoc can\'t import/find {0} {r1}, it reported error: "{2}", please check your spelling and sys.path'
            self.directive.warn(msg.format(self.objtype, str(self.fullname), err))
            self.env.note_reread()
            return False

    def add_directive_header(self, sig):
        """ Add the directive header 'attribute' with the annotation
        option set to the trait definition.

        """
        ClassLevelDocumenter.add_directive_header(self, sig)
        try:
            definition = trait_definition(cls=self.parent, trait_name=self.object_name)
        except ValueError:
            logger.warning('No definition for the trait {!r} could be found in class {!r}.'.format(self.object_name, self.parent), exc_info=True)
            return
        if '\n' in definition:
            definition = definition.partition('\n')[0] + ' …'
        self.add_line('   :annotation: = {0}'.format(definition), '<autodoc>')