from pyparsing import *
from sys import stdin, argv, exit
class MicroC(object):
    """Class for microC parser/compiler"""

    def __init__(self):
        self.tId = Word(alphas + '_', alphanums + '_')
        self.tInteger = Word(nums).setParseAction(lambda x: [x[0], SharedData.TYPES.INT])
        self.tUnsigned = Regex('[0-9]+[uU]').setParseAction(lambda x: [x[0][:-1], SharedData.TYPES.UNSIGNED])
        self.tConstant = (self.tUnsigned | self.tInteger).setParseAction(self.constant_action)
        self.tType = Keyword('int').setParseAction(lambda x: SharedData.TYPES.INT) | Keyword('unsigned').setParseAction(lambda x: SharedData.TYPES.UNSIGNED)
        self.tRelOp = oneOf(SharedData.RELATIONAL_OPERATORS)
        self.tMulOp = oneOf('* /')
        self.tAddOp = oneOf('+ -')
        self.rGlobalVariable = (self.tType('type') + self.tId('name') + FollowedBy(';')).setParseAction(self.global_variable_action)
        self.rGlobalVariableList = ZeroOrMore(self.rGlobalVariable + Suppress(';'))
        self.rExp = Forward()
        self.rMulExp = Forward()
        self.rNumExp = Forward()
        self.rArguments = delimitedList(self.rNumExp('exp').setParseAction(self.argument_action))
        self.rFunctionCall = ((self.tId('name') + FollowedBy('(')).setParseAction(self.function_call_prepare_action) + Suppress('(') + Optional(self.rArguments)('args') + Suppress(')')).setParseAction(self.function_call_action)
        self.rExp << (self.rFunctionCall | self.tConstant | self.tId('name').setParseAction(self.lookup_id_action) | Group(Suppress('(') + self.rNumExp + Suppress(')')) | Group('+' + self.rExp) | Group('-' + self.rExp)).setParseAction(lambda x: x[0])
        self.rMulExp << (self.rExp + ZeroOrMore(self.tMulOp + self.rExp)).setParseAction(self.mulexp_action)
        self.rNumExp << (self.rMulExp + ZeroOrMore(self.tAddOp + self.rMulExp)).setParseAction(self.numexp_action)
        self.rAndExp = Forward()
        self.rLogExp = Forward()
        self.rRelExp = (self.rNumExp + self.tRelOp + self.rNumExp).setParseAction(self.relexp_action)
        self.rAndExp << self.rRelExp('exp') + ZeroOrMore(Literal('&&').setParseAction(self.andexp_action) + self.rRelExp('exp')).setParseAction(lambda x: self.relexp_code)
        self.rLogExp << self.rAndExp('exp') + ZeroOrMore(Literal('||').setParseAction(self.logexp_action) + self.rAndExp('exp')).setParseAction(lambda x: self.andexp_code)
        self.rStatement = Forward()
        self.rStatementList = Forward()
        self.rReturnStatement = (Keyword('return') + self.rNumExp('exp') + Suppress(';')).setParseAction(self.return_action)
        self.rAssignmentStatement = (self.tId('var') + Suppress('=') + self.rNumExp('exp') + Suppress(';')).setParseAction(self.assignment_action)
        self.rFunctionCallStatement = self.rFunctionCall + Suppress(';')
        self.rIfStatement = ((Keyword('if') + FollowedBy('(')).setParseAction(self.if_begin_action) + (Suppress('(') + self.rLogExp + Suppress(')')).setParseAction(self.if_body_action) + (self.rStatement + Empty()).setParseAction(self.if_else_action) + Optional(Keyword('else') + self.rStatement)).setParseAction(self.if_end_action)
        self.rWhileStatement = ((Keyword('while') + FollowedBy('(')).setParseAction(self.while_begin_action) + (Suppress('(') + self.rLogExp + Suppress(')')).setParseAction(self.while_body_action) + self.rStatement).setParseAction(self.while_end_action)
        self.rCompoundStatement = Group(Suppress('{') + self.rStatementList + Suppress('}'))
        self.rStatement << (self.rReturnStatement | self.rIfStatement | self.rWhileStatement | self.rFunctionCallStatement | self.rAssignmentStatement | self.rCompoundStatement)
        self.rStatementList << ZeroOrMore(self.rStatement)
        self.rLocalVariable = (self.tType('type') + self.tId('name') + FollowedBy(';')).setParseAction(self.local_variable_action)
        self.rLocalVariableList = ZeroOrMore(self.rLocalVariable + Suppress(';'))
        self.rFunctionBody = Suppress('{') + Optional(self.rLocalVariableList).setParseAction(self.function_body_action) + self.rStatementList + Suppress('}')
        self.rParameter = (self.tType('type') + self.tId('name')).setParseAction(self.parameter_action)
        self.rParameterList = delimitedList(self.rParameter)
        self.rFunction = ((self.tType('type') + self.tId('name')).setParseAction(self.function_begin_action) + Group(Suppress('(') + Optional(self.rParameterList)('params') + Suppress(')') + self.rFunctionBody)).setParseAction(self.function_end_action)
        self.rFunctionList = OneOrMore(self.rFunction)
        self.rProgram = (Empty().setParseAction(self.data_begin_action) + self.rGlobalVariableList + Empty().setParseAction(self.code_begin_action) + self.rFunctionList).setParseAction(self.program_end_action)
        self.shared = SharedData()
        self.symtab = SymbolTable(self.shared)
        self.codegen = CodeGenerator(self.shared, self.symtab)
        self.function_call_index = -1
        self.function_call_stack = []
        self.function_arguments = []
        self.function_arguments_stack = []
        self.function_arguments_number = -1
        self.function_arguments_number_stack = []
        self.relexp_code = None
        self.andexp_code = None
        self.false_label_number = -1
        self.label_number = None
        self.label_stack = []

    def warning(self, message, print_location=True):
        """Displays warning message. Uses exshared for current location of parsing"""
        msg = 'Warning'
        if print_location and exshared.location != None:
            wline = lineno(exshared.location, exshared.text)
            wcol = col(exshared.location, exshared.text)
            wtext = line(exshared.location, exshared.text)
            msg += ' at line %d, col %d' % (wline, wcol)
        msg += ': %s' % message
        if print_location and exshared.location != None:
            msg += '\n%s' % wtext
        print(msg)

    def data_begin_action(self):
        """Inserts text at start of data segment"""
        self.codegen.prepare_data_segment()

    def code_begin_action(self):
        """Inserts text at start of code segment"""
        self.codegen.prepare_code_segment()

    def global_variable_action(self, text, loc, var):
        """Code executed after recognising a global variable"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('GLOBAL_VAR:', var)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        index = self.symtab.insert_global_var(var.name, var.type)
        self.codegen.global_var(var.name)
        return index

    def local_variable_action(self, text, loc, var):
        """Code executed after recognising a local variable"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('LOCAL_VAR:', var, var.name, var.type)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        index = self.symtab.insert_local_var(var.name, var.type, self.shared.function_vars)
        self.shared.function_vars += 1
        return index

    def parameter_action(self, text, loc, par):
        """Code executed after recognising a parameter"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('PARAM:', par)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        index = self.symtab.insert_parameter(par.name, par.type)
        self.shared.function_params += 1
        return index

    def constant_action(self, text, loc, const):
        """Code executed after recognising a constant"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('CONST:', const)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        return self.symtab.insert_constant(const[0], const[1])

    def function_begin_action(self, text, loc, fun):
        """Code executed after recognising a function definition (type and function name)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('FUN_BEGIN:', fun)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.shared.function_index = self.symtab.insert_function(fun.name, fun.type)
        self.shared.function_name = fun.name
        self.shared.function_params = 0
        self.shared.function_vars = 0
        self.codegen.function_begin()

    def function_body_action(self, text, loc, fun):
        """Code executed after recognising the beginning of function's body"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('FUN_BODY:', fun)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.codegen.function_body()

    def function_end_action(self, text, loc, fun):
        """Code executed at the end of function definition"""
        if DEBUG > 0:
            print('FUN_END:', fun)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.symtab.set_attribute(self.shared.function_index, self.shared.function_params)
        self.symtab.clear_symbols(self.shared.function_index + 1)
        self.codegen.function_end()

    def return_action(self, text, loc, ret):
        """Code executed after recognising a return statement"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('RETURN:', ret)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        if not self.symtab.same_types(self.shared.function_index, ret.exp[0]):
            raise SemanticException('Incompatible type in return')
        reg = self.codegen.take_function_register()
        self.codegen.move(ret.exp[0], reg)
        self.codegen.free_register(reg)
        self.codegen.unconditional_jump(self.codegen.label(self.shared.function_name + '_exit', True))

    def lookup_id_action(self, text, loc, var):
        """Code executed after recognising an identificator in expression"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('EXP_VAR:', var)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        var_index = self.symtab.lookup_symbol(var.name, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.PARAMETER, SharedData.KINDS.LOCAL_VAR])
        if var_index == None:
            raise SemanticException("'%s' undefined" % var.name)
        return var_index

    def assignment_action(self, text, loc, assign):
        """Code executed after recognising an assignment statement"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('ASSIGN:', assign)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        var_index = self.symtab.lookup_symbol(assign.var, [SharedData.KINDS.GLOBAL_VAR, SharedData.KINDS.PARAMETER, SharedData.KINDS.LOCAL_VAR])
        if var_index == None:
            raise SemanticException("Undefined lvalue '%s' in assignment" % assign.var)
        if not self.symtab.same_types(var_index, assign.exp[0]):
            raise SemanticException('Incompatible types in assignment')
        self.codegen.move(assign.exp[0], var_index)

    def mulexp_action(self, text, loc, mul):
        """Code executed after recognising a mulexp expression (something *|/ something)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('MUL_EXP:', mul)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        m = list(mul)
        while len(m) > 1:
            if not self.symtab.same_types(m[0], m[2]):
                raise SemanticException("Invalid opernads to binary '%s'" % m[1])
            reg = self.codegen.arithmetic(m[1], m[0], m[2])
            m[0:3] = [reg]
        return m[0]

    def numexp_action(self, text, loc, num):
        """Code executed after recognising a numexp expression (something +|- something)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('NUM_EXP:', num)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        n = list(num)
        while len(n) > 1:
            if not self.symtab.same_types(n[0], n[2]):
                raise SemanticException("Invalid opernads to binary '%s'" % n[1])
            reg = self.codegen.arithmetic(n[1], n[0], n[2])
            n[0:3] = [reg]
        return n[0]

    def function_call_prepare_action(self, text, loc, fun):
        """Code executed after recognising a function call (type and function name)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('FUN_PREP:', fun)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        index = self.symtab.lookup_symbol(fun.name, SharedData.KINDS.FUNCTION)
        if index == None:
            raise SemanticException("'%s' is not a function" % fun.name)
        self.function_call_stack.append(self.function_call_index)
        self.function_call_index = index
        self.function_arguments_stack.append(self.function_arguments[:])
        del self.function_arguments[:]
        self.codegen.save_used_registers()

    def argument_action(self, text, loc, arg):
        """Code executed after recognising each of function's arguments"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('ARGUMENT:', arg.exp)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        arg_ordinal = len(self.function_arguments)
        if not self.symtab.same_type_as_argument(arg.exp, self.function_call_index, arg_ordinal):
            raise SemanticException("Incompatible type for argument %d in '%s'" % (arg_ordinal + 1, self.symtab.get_name(self.function_call_index)))
        self.function_arguments.append(arg.exp)

    def function_call_action(self, text, loc, fun):
        """Code executed after recognising the whole function call"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('FUN_CALL:', fun)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        if len(self.function_arguments) != self.symtab.get_attribute(self.function_call_index):
            raise SemanticException("Wrong number of arguments for function '%s'" % fun.name)
        self.function_arguments.reverse()
        self.codegen.function_call(self.function_call_index, self.function_arguments)
        self.codegen.restore_used_registers()
        return_type = self.symtab.get_type(self.function_call_index)
        self.function_call_index = self.function_call_stack.pop()
        self.function_arguments = self.function_arguments_stack.pop()
        register = self.codegen.take_register(return_type)
        self.codegen.move(self.codegen.take_function_register(return_type), register)
        return register

    def relexp_action(self, text, loc, arg):
        """Code executed after recognising a relexp expression (something relop something)"""
        if DEBUG > 0:
            print('REL_EXP:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        exshared.setpos(loc, text)
        if not self.symtab.same_types(arg[0], arg[2]):
            raise SemanticException("Invalid operands for operator '{0}'".format(arg[1]))
        self.codegen.compare(arg[0], arg[2])
        self.relexp_code = self.codegen.relop_code(arg[1], self.symtab.get_type(arg[0]))
        return self.relexp_code

    def andexp_action(self, text, loc, arg):
        """Code executed after recognising a andexp expression (something and something)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('AND+EXP:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        label = self.codegen.label('false{0}'.format(self.false_label_number), True, False)
        self.codegen.jump(self.relexp_code, True, label)
        self.andexp_code = self.relexp_code
        return self.andexp_code

    def logexp_action(self, text, loc, arg):
        """Code executed after recognising logexp expression (something or something)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('LOG_EXP:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        label = self.codegen.label('true{0}'.format(self.label_number), True, False)
        self.codegen.jump(self.relexp_code, False, label)
        self.codegen.newline_label('false{0}'.format(self.false_label_number), True, True)
        self.false_label_number += 1

    def if_begin_action(self, text, loc, arg):
        """Code executed after recognising an if statement (if keyword)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('IF_BEGIN:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.false_label_number += 1
        self.label_number = self.false_label_number
        self.codegen.newline_label('if{0}'.format(self.label_number), True, True)

    def if_body_action(self, text, loc, arg):
        """Code executed after recognising if statement's body"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('IF_BODY:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        label = self.codegen.label('false{0}'.format(self.false_label_number), True, False)
        self.codegen.jump(self.relexp_code, True, label)
        self.codegen.newline_label('true{0}'.format(self.label_number), True, True)
        self.label_stack.append(self.false_label_number)
        self.label_stack.append(self.label_number)

    def if_else_action(self, text, loc, arg):
        """Code executed after recognising if statement's else body"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('IF_ELSE:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.label_number = self.label_stack.pop()
        label = self.codegen.label('exit{0}'.format(self.label_number), True, False)
        self.codegen.unconditional_jump(label)
        self.codegen.newline_label('false{0}'.format(self.label_stack.pop()), True, True)
        self.label_stack.append(self.label_number)

    def if_end_action(self, text, loc, arg):
        """Code executed after recognising a whole if statement"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('IF_END:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.codegen.newline_label('exit{0}'.format(self.label_stack.pop()), True, True)

    def while_begin_action(self, text, loc, arg):
        """Code executed after recognising a while statement (while keyword)"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('WHILE_BEGIN:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.false_label_number += 1
        self.label_number = self.false_label_number
        self.codegen.newline_label('while{0}'.format(self.label_number), True, True)

    def while_body_action(self, text, loc, arg):
        """Code executed after recognising while statement's body"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('WHILE_BODY:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        label = self.codegen.label('false{0}'.format(self.false_label_number), True, False)
        self.codegen.jump(self.relexp_code, True, label)
        self.codegen.newline_label('true{0}'.format(self.label_number), True, True)
        self.label_stack.append(self.false_label_number)
        self.label_stack.append(self.label_number)

    def while_end_action(self, text, loc, arg):
        """Code executed after recognising a whole while statement"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('WHILE_END:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        self.label_number = self.label_stack.pop()
        label = self.codegen.label('while{0}'.format(self.label_number), True, False)
        self.codegen.unconditional_jump(label)
        self.codegen.newline_label('false{0}'.format(self.label_stack.pop()), True, True)
        self.codegen.newline_label('exit{0}'.format(self.label_number), True, True)

    def program_end_action(self, text, loc, arg):
        """Checks if there is a 'main' function and the type of 'main' function"""
        exshared.setpos(loc, text)
        if DEBUG > 0:
            print('PROGRAM_END:', arg)
            if DEBUG == 2:
                self.symtab.display()
            if DEBUG > 2:
                return
        index = self.symtab.lookup_symbol('main', SharedData.KINDS.FUNCTION)
        if index == None:
            raise SemanticException("Undefined reference to 'main'", False)
        elif self.symtab.get_type(index) != SharedData.TYPES.INT:
            self.warning("Return type of 'main' is not int", False)

    def parse_text(self, text):
        """Parse string (helper function)"""
        try:
            return self.rProgram.ignore(cStyleComment).parseString(text, parseAll=True)
        except SemanticException as err:
            print(err)
            exit(3)
        except ParseException as err:
            print(err)
            exit(3)

    def parse_file(self, filename):
        """Parse file (helper function)"""
        try:
            return self.rProgram.ignore(cStyleComment).parseFile(filename, parseAll=True)
        except SemanticException as err:
            print(err)
            exit(3)
        except ParseException as err:
            print(err)
            exit(3)