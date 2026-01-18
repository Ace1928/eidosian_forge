from pyparsing import *
from sys import stdin, argv, exit
class SymbolTable(object):
    """Class for symbol table of microC program"""

    def __init__(self, shared):
        """Initialization of the symbol table"""
        self.table = []
        self.lable_len = 0
        for reg in range(SharedData.FUNCTION_REGISTER + 1):
            self.insert_symbol(SharedData.REGISTERS[reg], SharedData.KINDS.WORKING_REGISTER, SharedData.TYPES.NO_TYPE)
        self.shared = shared

    def error(self, text=''):
        """Symbol table error exception. It should happen only if index is out of range while accessing symbol table.
           This exeption is not handled by the compiler, so as to allow traceback printing
        """
        if text == '':
            raise Exception('Symbol table index out of range')
        else:
            raise Exception('Symbol table error: %s' % text)

    def display(self):
        """Displays the symbol table content"""
        sym_name = 'Symbol name'
        sym_len = max(max((len(i.name) for i in self.table)), len(sym_name))
        kind_name = 'Kind'
        kind_len = max(max((len(SharedData.KINDS[i.kind]) for i in self.table)), len(kind_name))
        type_name = 'Type'
        type_len = max(max((len(SharedData.TYPES[i.type]) for i in self.table)), len(type_name))
        attr_name = 'Attribute'
        attr_len = max(max((len(i.attribute_str()) for i in self.table)), len(attr_name))
        print('{0:3s} | {1:^{2}s} | {3:^{4}s} | {5:^{6}s} | {7:^{8}} | {9:s}'.format(' No', sym_name, sym_len, kind_name, kind_len, type_name, type_len, attr_name, attr_len, 'Parameters'))
        print('-----------------------------' + '-' * (sym_len + kind_len + type_len + attr_len))
        for i, sym in enumerate(self.table):
            parameters = ''
            for p in sym.param_types:
                if parameters == '':
                    parameters = '{0}'.format(SharedData.TYPES[p])
                else:
                    parameters += ', {0}'.format(SharedData.TYPES[p])
            print('{0:3d} | {1:^{2}s} | {3:^{4}s} | {5:^{6}s} | {7:^{8}} | ({9})'.format(i, sym.name, sym_len, SharedData.KINDS[sym.kind], kind_len, SharedData.TYPES[sym.type], type_len, sym.attribute_str(), attr_len, parameters))

    def insert_symbol(self, sname, skind, stype):
        """Inserts new symbol at the end of the symbol table.
           Returns symbol index
           sname - symbol name
           skind - symbol kind
           stype - symbol type
        """
        self.table.append(SymbolTableEntry(sname, skind, stype))
        self.table_len = len(self.table)
        return self.table_len - 1

    def clear_symbols(self, index):
        """Clears all symbols begining with the index to the end of table"""
        try:
            del self.table[index:]
        except Exception:
            self.error()
        self.table_len = len(self.table)

    def lookup_symbol(self, sname, skind=list(SharedData.KINDS.keys()), stype=list(SharedData.TYPES.keys())):
        """Searches for symbol, from the end to the begining.
           Returns symbol index or None
           sname - symbol name
           skind - symbol kind (one kind, list of kinds, or None) deafult: any kind
           stype - symbol type (or None) default: any type
        """
        skind = skind if isinstance(skind, list) else [skind]
        stype = stype if isinstance(stype, list) else [stype]
        for i, sym in [[x, self.table[x]] for x in range(len(self.table) - 1, SharedData.LAST_WORKING_REGISTER, -1)]:
            if sym.name == sname and sym.kind in skind and (sym.type in stype):
                return i
        return None

    def insert_id(self, sname, skind, skinds, stype):
        """Inserts a new identifier at the end of the symbol table, if possible.
           Returns symbol index, or raises an exception if the symbol alredy exists
           sname   - symbol name
           skind   - symbol kind
           skinds  - symbol kinds to check for
           stype   - symbol type
        """
        index = self.lookup_symbol(sname, skinds)
        if index == None:
            index = self.insert_symbol(sname, skind, stype)
            return index
        else:
            raise SemanticException("Redefinition of '%s'" % sname)

    def insert_global_var(self, vname, vtype):
        """Inserts a new global variable"""
        return self.insert_id(vname, SharedData.KINDS.GLOBAL_VAR, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.FUNCTION], vtype)

    def insert_local_var(self, vname, vtype, position):
        """Inserts a new local variable"""
        index = self.insert_id(vname, SharedData.KINDS.LOCAL_VAR, [SharedData.KINDS.LOCAL_VAR, SharedData.KINDS.PARAMETER], vtype)
        self.table[index].attribute = position

    def insert_parameter(self, pname, ptype):
        """Inserts a new parameter"""
        index = self.insert_id(pname, SharedData.KINDS.PARAMETER, SharedData.KINDS.PARAMETER, ptype)
        self.table[index].set_attribute('Index', self.shared.function_params)
        self.table[self.shared.function_index].param_types.append(ptype)
        return index

    def insert_function(self, fname, ftype):
        """Inserts a new function"""
        index = self.insert_id(fname, SharedData.KINDS.FUNCTION, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.FUNCTION], ftype)
        self.table[index].set_attribute('Params', 0)
        return index

    def insert_constant(self, cname, ctype):
        """Inserts a constant (or returns index if the constant already exists)
           Additionally, checks for range.
        """
        index = self.lookup_symbol(cname, stype=ctype)
        if index == None:
            num = int(cname)
            if ctype == SharedData.TYPES.INT:
                if num < SharedData.MIN_INT or num > SharedData.MAX_INT:
                    raise SemanticException("Integer constant '%s' out of range" % cname)
            elif ctype == SharedData.TYPES.UNSIGNED:
                if num < 0 or num > SharedData.MAX_UNSIGNED:
                    raise SemanticException("Unsigned constant '%s' out of range" % cname)
            index = self.insert_symbol(cname, SharedData.KINDS.CONSTANT, ctype)
        return index

    def same_types(self, index1, index2):
        """Returns True if both symbol table elements are of the same type"""
        try:
            same = self.table[index1].type == self.table[index2].type != SharedData.TYPES.NO_TYPE
        except Exception:
            self.error()
        return same

    def same_type_as_argument(self, index, function_index, argument_number):
        """Returns True if index and function's argument are of the same type
           index - index in symbol table
           function_index - function's index in symbol table
           argument_number - # of function's argument
        """
        try:
            same = self.table[function_index].param_types[argument_number] == self.table[index].type
        except Exception:
            self.error()
        return same

    def get_attribute(self, index):
        try:
            return self.table[index].attribute
        except Exception:
            self.error()

    def set_attribute(self, index, value):
        try:
            self.table[index].attribute = value
        except Exception:
            self.error()

    def get_name(self, index):
        try:
            return self.table[index].name
        except Exception:
            self.error()

    def get_kind(self, index):
        try:
            return self.table[index].kind
        except Exception:
            self.error()

    def get_type(self, index):
        try:
            return self.table[index].type
        except Exception:
            self.error()

    def set_type(self, index, stype):
        try:
            self.table[index].type = stype
        except Exception:
            self.error()