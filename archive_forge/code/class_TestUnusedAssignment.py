from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
class TestUnusedAssignment(TestCase):
    """
    Tests for warning about unused assignments.
    """

    def test_unusedVariable(self):
        """
        Warn when a variable in a function is assigned a value that's never
        used.
        """
        self.flakes('\n        def a():\n            b = 1\n        ', m.UnusedVariable)

    def test_unusedUnderscoreVariable(self):
        """
        Don't warn when the magic "_" (underscore) variable is unused.
        See issue #202.
        """
        self.flakes('\n        def a(unused_param):\n            _ = unused_param\n        ')

    def test_unusedVariableAsLocals(self):
        """
        Using locals() it is perfectly valid to have unused variables
        """
        self.flakes('\n        def a():\n            b = 1\n            return locals()\n        ')

    def test_unusedVariableNoLocals(self):
        """
        Using locals() in wrong scope should not matter
        """
        self.flakes('\n        def a():\n            locals()\n            def a():\n                b = 1\n                return\n        ', m.UnusedVariable)

    @skip("todo: Difficult because it doesn't apply in the context of a loop")
    def test_unusedReassignedVariable(self):
        """
        Shadowing a used variable can still raise an UnusedVariable warning.
        """
        self.flakes('\n        def a():\n            b = 1\n            b.foo()\n            b = 2\n        ', m.UnusedVariable)

    def test_variableUsedInLoop(self):
        """
        Shadowing a used variable cannot raise an UnusedVariable warning in the
        context of a loop.
        """
        self.flakes('\n        def a():\n            b = True\n            while b:\n                b = False\n        ')

    def test_assignToGlobal(self):
        """
        Assigning to a global and then not using that global is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
        self.flakes('\n        b = 0\n        def a():\n            global b\n            b = 1\n        ')

    def test_assignToNonlocal(self):
        """
        Assigning to a nonlocal and then not using that binding is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
        self.flakes("\n        b = b'0'\n        def a():\n            nonlocal b\n            b = b'1'\n        ")

    def test_assignToMember(self):
        """
        Assigning to a member of another object and then not using that member
        variable is perfectly acceptable. Do not mistake it for an unused
        local variable.
        """
        self.flakes('\n        class b:\n            pass\n        def a():\n            b.foo = 1\n        ')

    def test_assignInForLoop(self):
        """
        Don't warn when a variable in a for loop is assigned to but not used.
        """
        self.flakes('\n        def f():\n            for i in range(10):\n                pass\n        ')

    def test_assignInListComprehension(self):
        """
        Don't warn when a variable in a list comprehension is
        assigned to but not used.
        """
        self.flakes('\n        def f():\n            [None for i in range(10)]\n        ')

    def test_generatorExpression(self):
        """
        Don't warn when a variable in a generator expression is
        assigned to but not used.
        """
        self.flakes('\n        def f():\n            (None for i in range(10))\n        ')

    def test_assignmentInsideLoop(self):
        """
        Don't warn when a variable assignment occurs lexically after its use.
        """
        self.flakes('\n        def f():\n            x = None\n            for i in range(10):\n                if i > 2:\n                    return x\n                x = i * 2\n        ')

    def test_tupleUnpacking(self):
        """
        Don't warn when a variable included in tuple unpacking is unused. It's
        very common for variables in a tuple unpacking assignment to be unused
        in good Python code, so warning will only create false positives.
        """
        self.flakes('\n        def f(tup):\n            (x, y) = tup\n        ')
        self.flakes('\n        def f():\n            (x, y) = 1, 2\n        ', m.UnusedVariable, m.UnusedVariable)
        self.flakes('\n        def f():\n            (x, y) = coords = 1, 2\n            if x > 1:\n                print(coords)\n        ')
        self.flakes('\n        def f():\n            (x, y) = coords = 1, 2\n        ', m.UnusedVariable)
        self.flakes('\n        def f():\n            coords = (x, y) = 1, 2\n        ', m.UnusedVariable)

    def test_listUnpacking(self):
        """
        Don't warn when a variable included in list unpacking is unused.
        """
        self.flakes('\n        def f(tup):\n            [x, y] = tup\n        ')
        self.flakes('\n        def f():\n            [x, y] = [1, 2]\n        ', m.UnusedVariable, m.UnusedVariable)

    def test_closedOver(self):
        """
        Don't warn when the assignment is used in an inner function.
        """
        self.flakes('\n        def barMaker():\n            foo = 5\n            def bar():\n                return foo\n            return bar\n        ')

    def test_doubleClosedOver(self):
        """
        Don't warn when the assignment is used in an inner function, even if
        that inner function itself is in an inner function.
        """
        self.flakes('\n        def barMaker():\n            foo = 5\n            def bar():\n                def baz():\n                    return foo\n            return bar\n        ')

    def test_tracebackhideSpecialVariable(self):
        """
        Do not warn about unused local variable __tracebackhide__, which is
        a special variable for py.test.
        """
        self.flakes('\n            def helper():\n                __tracebackhide__ = True\n        ')

    def test_ifexp(self):
        """
        Test C{foo if bar else baz} statements.
        """
        self.flakes("a = 'moo' if True else 'oink'")
        self.flakes("a = foo if True else 'oink'", m.UndefinedName)
        self.flakes("a = 'moo' if True else bar", m.UndefinedName)

    def test_if_tuple(self):
        """
        Test C{if (foo,)} conditions.
        """
        self.flakes('if (): pass')
        self.flakes('\n        if (\n            True\n        ):\n            pass\n        ')
        self.flakes('\n        if (\n            True,\n        ):\n            pass\n        ', m.IfTuple)
        self.flakes('\n        x = 1 if (\n            True,\n        ) else 2\n        ', m.IfTuple)

    def test_withStatementNoNames(self):
        """
        No warnings are emitted for using inside or after a nameless C{with}
        statement a name defined beforehand.
        """
        self.flakes('\n        bar = None\n        with open("foo"):\n            bar\n        bar\n        ')

    def test_withStatementSingleName(self):
        """
        No warnings are emitted for using a name defined by a C{with} statement
        within the suite or afterwards.
        """
        self.flakes("\n        with open('foo') as bar:\n            bar\n        bar\n        ")

    def test_withStatementAttributeName(self):
        """
        No warnings are emitted for using an attribute as the target of a
        C{with} statement.
        """
        self.flakes("\n        import foo\n        with open('foo') as foo.bar:\n            pass\n        ")

    def test_withStatementSubscript(self):
        """
        No warnings are emitted for using a subscript as the target of a
        C{with} statement.
        """
        self.flakes("\n        import foo\n        with open('foo') as foo[0]:\n            pass\n        ")

    def test_withStatementSubscriptUndefined(self):
        """
        An undefined name warning is emitted if the subscript used as the
        target of a C{with} statement is not defined.
        """
        self.flakes("\n        import foo\n        with open('foo') as foo[bar]:\n            pass\n        ", m.UndefinedName)

    def test_withStatementTupleNames(self):
        """
        No warnings are emitted for using any of the tuple of names defined by
        a C{with} statement within the suite or afterwards.
        """
        self.flakes("\n        with open('foo') as (bar, baz):\n            bar, baz\n        bar, baz\n        ")

    def test_withStatementListNames(self):
        """
        No warnings are emitted for using any of the list of names defined by a
        C{with} statement within the suite or afterwards.
        """
        self.flakes("\n        with open('foo') as [bar, baz]:\n            bar, baz\n        bar, baz\n        ")

    def test_withStatementComplicatedTarget(self):
        """
        If the target of a C{with} statement uses any or all of the valid forms
        for that part of the grammar (See
        U{http://docs.python.org/reference/compound_stmts.html#the-with-statement}),
        the names involved are checked both for definedness and any bindings
        created are respected in the suite of the statement and afterwards.
        """
        self.flakes("\n        c = d = e = g = h = i = None\n        with open('foo') as [(a, b), c[d], e.f, g[h:i]]:\n            a, b, c, d, e, g, h, i\n        a, b, c, d, e, g, h, i\n        ")

    def test_withStatementSingleNameUndefined(self):
        """
        An undefined name warning is emitted if the name first defined by a
        C{with} statement is used before the C{with} statement.
        """
        self.flakes("\n        bar\n        with open('foo') as bar:\n            pass\n        ", m.UndefinedName)

    def test_withStatementTupleNamesUndefined(self):
        """
        An undefined name warning is emitted if a name first defined by the
        tuple-unpacking form of the C{with} statement is used before the
        C{with} statement.
        """
        self.flakes("\n        baz\n        with open('foo') as (bar, baz):\n            pass\n        ", m.UndefinedName)

    def test_withStatementSingleNameRedefined(self):
        """
        A redefined name warning is emitted if a name bound by an import is
        rebound by the name defined by a C{with} statement.
        """
        self.flakes("\n        import bar\n        with open('foo') as bar:\n            pass\n        ", m.RedefinedWhileUnused)

    def test_withStatementTupleNamesRedefined(self):
        """
        A redefined name warning is emitted if a name bound by an import is
        rebound by one of the names defined by the tuple-unpacking form of a
        C{with} statement.
        """
        self.flakes("\n        import bar\n        with open('foo') as (bar, baz):\n            pass\n        ", m.RedefinedWhileUnused)

    def test_withStatementUndefinedInside(self):
        """
        An undefined name warning is emitted if a name is used inside the
        body of a C{with} statement without first being bound.
        """
        self.flakes("\n        with open('foo') as bar:\n            baz\n        ", m.UndefinedName)

    def test_withStatementNameDefinedInBody(self):
        """
        A name defined in the body of a C{with} statement can be used after
        the body ends without warning.
        """
        self.flakes("\n        with open('foo') as bar:\n            baz = 10\n        baz\n        ")

    def test_withStatementUndefinedInExpression(self):
        """
        An undefined name warning is emitted if a name in the I{test}
        expression of a C{with} statement is undefined.
        """
        self.flakes('\n        with bar as baz:\n            pass\n        ', m.UndefinedName)
        self.flakes('\n        with bar as bar:\n            pass\n        ', m.UndefinedName)

    def test_dictComprehension(self):
        """
        Dict comprehensions are properly handled.
        """
        self.flakes('\n        a = {1: x for x in range(10)}\n        ')

    def test_setComprehensionAndLiteral(self):
        """
        Set comprehensions are properly handled.
        """
        self.flakes('\n        a = {1, 2, 3}\n        b = {x for x in range(10)}\n        ')

    def test_exceptionUsedInExcept(self):
        self.flakes('\n        try: pass\n        except Exception as e: e\n        ')
        self.flakes('\n        def download_review():\n            try: pass\n            except Exception as e: e\n        ')

    def test_exceptionUnusedInExcept(self):
        self.flakes('\n        try: pass\n        except Exception as e: pass\n        ', m.UnusedVariable)

    @skipIf(version_info < (3, 11), 'new in Python 3.11')
    def test_exception_unused_in_except_star(self):
        self.flakes('\n            try:\n                pass\n            except* OSError as e:\n                pass\n        ', m.UnusedVariable)

    def test_exceptionUnusedInExceptInFunction(self):
        self.flakes('\n        def download_review():\n            try: pass\n            except Exception as e: pass\n        ', m.UnusedVariable)

    def test_exceptWithoutNameInFunction(self):
        """
        Don't issue false warning when an unnamed exception is used.
        Previously, there would be a false warning, but only when the
        try..except was in a function
        """
        self.flakes('\n        import tokenize\n        def foo():\n            try: pass\n            except tokenize.TokenError: pass\n        ')

    def test_exceptWithoutNameInFunctionTuple(self):
        """
        Don't issue false warning when an unnamed exception is used.
        This example catches a tuple of exception types.
        """
        self.flakes('\n        import tokenize\n        def foo():\n            try: pass\n            except (tokenize.TokenError, IndentationError): pass\n        ')

    def test_augmentedAssignmentImportedFunctionCall(self):
        """
        Consider a function that is called on the right part of an
        augassign operation to be used.
        """
        self.flakes('\n        from foo import bar\n        baz = 0\n        baz += bar()\n        ')

    def test_assert_without_message(self):
        """An assert without a message is not an error."""
        self.flakes('\n        a = 1\n        assert a\n        ')

    def test_assert_with_message(self):
        """An assert with a message is not an error."""
        self.flakes("\n        a = 1\n        assert a, 'x'\n        ")

    def test_assert_tuple(self):
        """An assert of a non-empty tuple is always True."""
        self.flakes("\n        assert (False, 'x')\n        assert (False, )\n        ", m.AssertTuple, m.AssertTuple)

    def test_assert_tuple_empty(self):
        """An assert of an empty tuple is always False."""
        self.flakes('\n        assert ()\n        ')

    def test_assert_static(self):
        """An assert of a static value is not an error."""
        self.flakes('\n        assert True\n        assert 1\n        ')

    def test_yieldFromUndefined(self):
        """
        Test C{yield from} statement
        """
        self.flakes('\n        def bar():\n            yield from foo()\n        ', m.UndefinedName)

    def test_f_string(self):
        """Test PEP 498 f-strings are treated as a usage."""
        self.flakes("\n        baz = 0\n        print(f'{4*baz}')\n        ")

    def test_assign_expr(self):
        """Test PEP 572 assignment expressions are treated as usage / write."""
        self.flakes('\n        from foo import y\n        print(x := y)\n        print(x)\n        ')

    def test_assign_expr_generator_scope(self):
        """Test assignment expressions in generator expressions."""
        self.flakes('\n        if (any((y := x[0]) for x in [[True]])):\n            print(y)\n        ')

    def test_assign_expr_nested(self):
        """Test assignment expressions in nested expressions."""
        self.flakes('\n        if ([(y:=x) for x in range(4) if [(z:=q) for q in range(4)]]):\n            print(y)\n            print(z)\n        ')