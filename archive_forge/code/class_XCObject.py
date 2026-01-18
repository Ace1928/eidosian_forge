import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCObject:
    """The abstract base of all class types used in Xcode project files.

  Class variables:
    _schema: A dictionary defining the properties of this class.  The keys to
             _schema are string property keys as used in project files.  Values
             are a list of four or five elements:
             [ is_list, property_type, is_strong, is_required, default ]
             is_list: True if the property described is a list, as opposed
                      to a single element.
             property_type: The type to use as the value of the property,
                            or if is_list is True, the type to use for each
                            element of the value's list.  property_type must
                            be an XCObject subclass, or one of the built-in
                            types str, int, or dict.
             is_strong: If property_type is an XCObject subclass, is_strong
                        is True to assert that this class "owns," or serves
                        as parent, to the property value (or, if is_list is
                        True, values).  is_strong must be False if
                        property_type is not an XCObject subclass.
             is_required: True if the property is required for the class.
                          Note that is_required being True does not preclude
                          an empty string ("", in the case of property_type
                          str) or list ([], in the case of is_list True) from
                          being set for the property.
             default: Optional.  If is_required is True, default may be set
                      to provide a default value for objects that do not supply
                      their own value.  If is_required is True and default
                      is not provided, users of the class must supply their own
                      value for the property.
             Note that although the values of the array are expressed in
             boolean terms, subclasses provide values as integers to conserve
             horizontal space.
    _should_print_single_line: False in XCObject.  Subclasses whose objects
                               should be written to the project file in the
                               alternate single-line format, such as
                               PBXFileReference and PBXBuildFile, should
                               set this to True.
    _encode_transforms: Used by _EncodeString to encode unprintable characters.
                        The index into this list is the ordinal of the
                        character to transform; each value is a string
                        used to represent the character in the output.  XCObject
                        provides an _encode_transforms list suitable for most
                        XCObject subclasses.
    _alternate_encode_transforms: Provided for subclasses that wish to use
                                  the alternate encoding rules.  Xcode seems
                                  to use these rules when printing objects in
                                  single-line format.  Subclasses that desire
                                  this behavior should set _encode_transforms
                                  to _alternate_encode_transforms.
    _hashables: A list of XCObject subclasses that can be hashed by ComputeIDs
                to construct this object's ID.  Most classes that need custom
                hashing behavior should do it by overriding Hashables,
                but in some cases an object's parent may wish to push a
                hashable value into its child, and it can do so by appending
                to _hashables.
  Attributes:
    id: The object's identifier, a 24-character uppercase hexadecimal string.
        Usually, objects being created should not set id until the entire
        project file structure is built.  At that point, UpdateIDs() should
        be called on the root object to assign deterministic values for id to
        each object in the tree.
    parent: The object's parent.  This is set by a parent XCObject when a child
            object is added to it.
    _properties: The object's property dictionary.  An object's properties are
                 described by its class' _schema variable.
  """
    _schema = {}
    _should_print_single_line = False
    _encode_transforms = []
    i = 0
    while i < ord(' '):
        _encode_transforms.append('\\U%04x' % i)
        i = i + 1
    _encode_transforms[7] = '\\a'
    _encode_transforms[8] = '\\b'
    _encode_transforms[9] = '\\t'
    _encode_transforms[10] = '\\n'
    _encode_transforms[11] = '\\v'
    _encode_transforms[12] = '\\f'
    _encode_transforms[13] = '\\n'
    _alternate_encode_transforms = list(_encode_transforms)
    _alternate_encode_transforms[9] = chr(9)
    _alternate_encode_transforms[10] = chr(10)
    _alternate_encode_transforms[11] = chr(11)

    def __init__(self, properties=None, id=None, parent=None):
        self.id = id
        self.parent = parent
        self._properties = {}
        self._hashables = []
        self._SetDefaultsFromSchema()
        self.UpdateProperties(properties)

    def __repr__(self):
        try:
            name = self.Name()
        except NotImplementedError:
            return f'<{self.__class__.__name__} at 0x{id(self):x}>'
        return f'<{self.__class__.__name__} {name!r} at 0x{id(self):x}>'

    def Copy(self):
        """Make a copy of this object.

    The new object will have its own copy of lists and dicts.  Any XCObject
    objects owned by this object (marked "strong") will be copied in the
    new object, even those found in lists.  If this object has any weak
    references to other XCObjects, the same references are added to the new
    object without making a copy.
    """
        that = self.__class__(id=self.id, parent=self.parent)
        for key, value in self._properties.items():
            is_strong = self._schema[key][2]
            if isinstance(value, XCObject):
                if is_strong:
                    new_value = value.Copy()
                    new_value.parent = that
                    that._properties[key] = new_value
                else:
                    that._properties[key] = value
            elif isinstance(value, (str, int)):
                that._properties[key] = value
            elif isinstance(value, list):
                if is_strong:
                    that._properties[key] = []
                    for item in value:
                        new_item = item.Copy()
                        new_item.parent = that
                        that._properties[key].append(new_item)
                else:
                    that._properties[key] = value[:]
            elif isinstance(value, dict):
                if is_strong:
                    raise TypeError('Strong dict for key ' + key + ' in ' + self.__class__.__name__)
                else:
                    that._properties[key] = value.copy()
            else:
                raise TypeError('Unexpected type ' + value.__class__.__name__ + ' for key ' + key + ' in ' + self.__class__.__name__)
        return that

    def Name(self):
        """Return the name corresponding to an object.

    Not all objects necessarily need to be nameable, and not all that do have
    a "name" property.  Override as needed.
    """
        if 'name' in self._properties or ('name' in self._schema and self._schema['name'][3]):
            return self._properties['name']
        raise NotImplementedError(self.__class__.__name__ + ' must implement Name')

    def Comment(self):
        """Return a comment string for the object.

    Most objects just use their name as the comment, but PBXProject uses
    different values.

    The returned comment is not escaped and does not have any comment marker
    strings applied to it.
    """
        return self.Name()

    def Hashables(self):
        hashables = [self.__class__.__name__]
        name = self.Name()
        if name is not None:
            hashables.append(name)
        hashables.extend(self._hashables)
        return hashables

    def HashablesForChild(self):
        return None

    def ComputeIDs(self, recursive=True, overwrite=True, seed_hash=None):
        """Set "id" properties deterministically.

    An object's "id" property is set based on a hash of its class type and
    name, as well as the class type and name of all ancestor objects.  As
    such, it is only advisable to call ComputeIDs once an entire project file
    tree is built.

    If recursive is True, recurse into all descendant objects and update their
    hashes.

    If overwrite is True, any existing value set in the "id" property will be
    replaced.
    """

        def _HashUpdate(hash, data):
            """Update hash with data's length and contents.

      If the hash were updated only with the value of data, it would be
      possible for clowns to induce collisions by manipulating the names of
      their objects.  By adding the length, it's exceedingly less likely that
      ID collisions will be encountered, intentionally or not.
      """
            hash.update(struct.pack('>i', len(data)))
            if isinstance(data, str):
                data = data.encode('utf-8')
            hash.update(data)
        if seed_hash is None:
            seed_hash = hashlib.sha1()
        hash = seed_hash.copy()
        hashables = self.Hashables()
        assert len(hashables) > 0
        for hashable in hashables:
            _HashUpdate(hash, hashable)
        if recursive:
            hashables_for_child = self.HashablesForChild()
            if hashables_for_child is None:
                child_hash = hash
            else:
                assert len(hashables_for_child) > 0
                child_hash = seed_hash.copy()
                for hashable in hashables_for_child:
                    _HashUpdate(child_hash, hashable)
            for child in self.Children():
                child.ComputeIDs(recursive, overwrite, child_hash)
        if overwrite or self.id is None:
            assert hash.digest_size % 4 == 0
            digest_int_count = hash.digest_size // 4
            digest_ints = struct.unpack('>' + 'I' * digest_int_count, hash.digest())
            id_ints = [0, 0, 0]
            for index in range(0, digest_int_count):
                id_ints[index % 3] ^= digest_ints[index]
            self.id = '%08X%08X%08X' % tuple(id_ints)

    def EnsureNoIDCollisions(self):
        """Verifies that no two objects have the same ID.  Checks all descendants.
    """
        ids = {}
        descendants = self.Descendants()
        for descendant in descendants:
            if descendant.id in ids:
                other = ids[descendant.id]
                raise KeyError('Duplicate ID %s, objects "%s" and "%s" in "%s"' % (descendant.id, str(descendant._properties), str(other._properties), self._properties['rootObject'].Name()))
            ids[descendant.id] = descendant

    def Children(self):
        """Returns a list of all of this object's owned (strong) children."""
        children = []
        for property, attributes in self._schema.items():
            is_list, property_type, is_strong = attributes[0:3]
            if is_strong and property in self._properties:
                if not is_list:
                    children.append(self._properties[property])
                else:
                    children.extend(self._properties[property])
        return children

    def Descendants(self):
        """Returns a list of all of this object's descendants, including this
    object.
    """
        children = self.Children()
        descendants = [self]
        for child in children:
            descendants.extend(child.Descendants())
        return descendants

    def PBXProjectAncestor(self):
        if self.parent:
            return self.parent.PBXProjectAncestor()
        return None

    def _EncodeComment(self, comment):
        """Encodes a comment to be placed in the project file output, mimicking
    Xcode behavior.
    """
        return '/* ' + comment.replace('*/', '(*)/') + ' */'

    def _EncodeTransform(self, match):
        char = match.group(0)
        if char == '\\':
            return '\\\\'
        if char == '"':
            return '\\"'
        return self._encode_transforms[ord(char)]

    def _EncodeString(self, value):
        """Encodes a string to be placed in the project file output, mimicking
    Xcode behavior.
    """
        if _unquoted.search(value) and (not _quoted.search(value)):
            return value
        return '"' + _escaped.sub(self._EncodeTransform, value) + '"'

    def _XCPrint(self, file, tabs, line):
        file.write('\t' * tabs + line)

    def _XCPrintableValue(self, tabs, value, flatten_list=False):
        """Returns a representation of value that may be printed in a project file,
    mimicking Xcode's behavior.

    _XCPrintableValue can handle str and int values, XCObjects (which are
    made printable by returning their id property), and list and dict objects
    composed of any of the above types.  When printing a list or dict, and
    _should_print_single_line is False, the tabs parameter is used to determine
    how much to indent the lines corresponding to the items in the list or
    dict.

    If flatten_list is True, single-element lists will be transformed into
    strings.
    """
        printable = ''
        comment = None
        if self._should_print_single_line:
            sep = ' '
            element_tabs = ''
            end_tabs = ''
        else:
            sep = '\n'
            element_tabs = '\t' * (tabs + 1)
            end_tabs = '\t' * tabs
        if isinstance(value, XCObject):
            printable += value.id
            comment = value.Comment()
        elif isinstance(value, str):
            printable += self._EncodeString(value)
        elif isinstance(value, str):
            printable += self._EncodeString(value.encode('utf-8'))
        elif isinstance(value, int):
            printable += str(value)
        elif isinstance(value, list):
            if flatten_list and len(value) <= 1:
                if len(value) == 0:
                    printable += self._EncodeString('')
                else:
                    printable += self._EncodeString(value[0])
            else:
                printable = '(' + sep
                for item in value:
                    printable += element_tabs + self._XCPrintableValue(tabs + 1, item, flatten_list) + ',' + sep
                printable += end_tabs + ')'
        elif isinstance(value, dict):
            printable = '{' + sep
            for item_key, item_value in sorted(value.items()):
                printable += element_tabs + self._XCPrintableValue(tabs + 1, item_key, flatten_list) + ' = ' + self._XCPrintableValue(tabs + 1, item_value, flatten_list) + ';' + sep
            printable += end_tabs + '}'
        else:
            raise TypeError("Can't make " + value.__class__.__name__ + ' printable')
        if comment:
            printable += ' ' + self._EncodeComment(comment)
        return printable

    def _XCKVPrint(self, file, tabs, key, value):
        """Prints a key and value, members of an XCObject's _properties dictionary,
    to file.

    tabs is an int identifying the indentation level.  If the class'
    _should_print_single_line variable is True, tabs is ignored and the
    key-value pair will be followed by a space insead of a newline.
    """
        if self._should_print_single_line:
            printable = ''
            after_kv = ' '
        else:
            printable = '\t' * tabs
            after_kv = '\n'
        if key == 'remoteGlobalIDString' and isinstance(self, PBXContainerItemProxy):
            value_to_print = value.id
        else:
            value_to_print = value
        if key == 'settings' and isinstance(self, PBXBuildFile):
            strip_value_quotes = True
        else:
            strip_value_quotes = False
        if key == 'buildSettings' and isinstance(self, XCBuildConfiguration):
            flatten_list = True
        else:
            flatten_list = False
        try:
            printable_key = self._XCPrintableValue(tabs, key, flatten_list)
            printable_value = self._XCPrintableValue(tabs, value_to_print, flatten_list)
            if strip_value_quotes and len(printable_value) > 1 and (printable_value[0] == '"') and (printable_value[-1] == '"'):
                printable_value = printable_value[1:-1]
            printable += printable_key + ' = ' + printable_value + ';' + after_kv
        except TypeError as e:
            gyp.common.ExceptionAppend(e, 'while printing key "%s"' % key)
            raise
        self._XCPrint(file, 0, printable)

    def Print(self, file=sys.stdout):
        """Prints a reprentation of this object to file, adhering to Xcode output
    formatting.
    """
        self.VerifyHasRequiredProperties()
        if self._should_print_single_line:
            sep = ''
            end_tabs = 0
        else:
            sep = '\n'
            end_tabs = 2
        self._XCPrint(file, 2, self._XCPrintableValue(2, self) + ' = {' + sep)
        self._XCKVPrint(file, 3, 'isa', self.__class__.__name__)
        for property, value in sorted(self._properties.items()):
            self._XCKVPrint(file, 3, property, value)
        self._XCPrint(file, end_tabs, '};\n')

    def UpdateProperties(self, properties, do_copy=False):
        """Merge the supplied properties into the _properties dictionary.

    The input properties must adhere to the class schema or a KeyError or
    TypeError exception will be raised.  If adding an object of an XCObject
    subclass and the schema indicates a strong relationship, the object's
    parent will be set to this object.

    If do_copy is True, then lists, dicts, strong-owned XCObjects, and
    strong-owned XCObjects in lists will be copied instead of having their
    references added.
    """
        if properties is None:
            return
        for property, value in properties.items():
            if property not in self._schema:
                raise KeyError(property + ' not in ' + self.__class__.__name__)
            is_list, property_type, is_strong = self._schema[property][0:3]
            if is_list:
                if value.__class__ != list:
                    raise TypeError(property + ' of ' + self.__class__.__name__ + ' must be list, not ' + value.__class__.__name__)
                for item in value:
                    if not isinstance(item, property_type) and (not (isinstance(item, str) and property_type == str)):
                        raise TypeError('item of ' + property + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + item.__class__.__name__)
            elif not isinstance(value, property_type) and (not (isinstance(value, str) and property_type == str)):
                raise TypeError(property + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + value.__class__.__name__)
            if do_copy:
                if isinstance(value, XCObject):
                    if is_strong:
                        self._properties[property] = value.Copy()
                    else:
                        self._properties[property] = value
                elif isinstance(value, (str, int)):
                    self._properties[property] = value
                elif isinstance(value, list):
                    if is_strong:
                        self._properties[property] = []
                        for item in value:
                            self._properties[property].append(item.Copy())
                    else:
                        self._properties[property] = value[:]
                elif isinstance(value, dict):
                    self._properties[property] = value.copy()
                else:
                    raise TypeError("Don't know how to copy a " + value.__class__.__name__ + ' object for ' + property + ' in ' + self.__class__.__name__)
            else:
                self._properties[property] = value
            if is_strong:
                if not is_list:
                    self._properties[property].parent = self
                else:
                    for item in self._properties[property]:
                        item.parent = self

    def HasProperty(self, key):
        return key in self._properties

    def GetProperty(self, key):
        return self._properties[key]

    def SetProperty(self, key, value):
        self.UpdateProperties({key: value})

    def DelProperty(self, key):
        if key in self._properties:
            del self._properties[key]

    def AppendProperty(self, key, value):
        if key not in self._schema:
            raise KeyError(key + ' not in ' + self.__class__.__name__)
        is_list, property_type, is_strong = self._schema[key][0:3]
        if not is_list:
            raise TypeError(key + ' of ' + self.__class__.__name__ + ' must be list')
        if not isinstance(value, property_type):
            raise TypeError('item of ' + key + ' of ' + self.__class__.__name__ + ' must be ' + property_type.__name__ + ', not ' + value.__class__.__name__)
        self._properties[key] = self._properties.get(key, [])
        if is_strong:
            value.parent = self
        self._properties[key].append(value)

    def VerifyHasRequiredProperties(self):
        """Ensure that all properties identified as required by the schema are
    set.
    """
        for property, attributes in self._schema.items():
            is_list, property_type, is_strong, is_required = attributes[0:4]
            if is_required and property not in self._properties:
                raise KeyError(self.__class__.__name__ + ' requires ' + property)

    def _SetDefaultsFromSchema(self):
        """Assign object default values according to the schema.  This will not
    overwrite properties that have already been set."""
        defaults = {}
        for property, attributes in self._schema.items():
            is_list, property_type, is_strong, is_required = attributes[0:4]
            if is_required and len(attributes) >= 5 and (property not in self._properties):
                default = attributes[4]
                defaults[property] = default
        if len(defaults) > 0:
            self.UpdateProperties(defaults, do_copy=True)