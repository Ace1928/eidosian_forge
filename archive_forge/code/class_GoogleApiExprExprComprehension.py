from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprComprehension(_messages.Message):
    """A comprehension expression applied to a list or map. Comprehensions are
  not part of the core syntax, but enabled with macros. A macro matches a
  specific call signature within a parsed AST and replaces the call with an
  alternate AST block. Macro expansion happens at parse time. The following
  macros are supported within CEL: Aggregate type macros may be applied to all
  elements in a list or all keys in a map: * `all`, `exists`, `exists_one` -
  test a predicate expression against the inputs and return `true` if the
  predicate is satisfied for all, any, or only one value `list.all(x, x <
  10)`. * `filter` - test a predicate expression against the inputs and return
  the subset of elements which satisfy the predicate: `payments.filter(p, p >
  1000)`. * `map` - apply an expression to all elements in the input and
  return the output aggregate type: `[1, 2, 3].map(i, i * i)`. The `has(m.x)`
  macro tests whether the property `x` is present in struct `m`. The semantics
  of this macro depend on the type of `m`. For proto2 messages `has(m.x)` is
  defined as 'defined, but not set`. For proto3, the macro tests whether the
  property is set to its default. For map and struct types, the macro tests
  whether the property `x` is defined on `m`. Comprehension evaluation can be
  best visualized as the following pseudocode: ``` let `accu_var` =
  `accu_init` for (let `iter_var` in `iter_range`) { if (!`loop_condition`) {
  break } `accu_var` = `loop_step` } return `result` ```

  Fields:
    accuInit: The initial value of the accumulator.
    accuVar: The name of the variable used for accumulation of the result.
    iterRange: The range over which var iterates.
    iterVar: The name of the iteration variable.
    loopCondition: An expression which can contain iter_var and accu_var.
      Returns false when the result has been computed and may be used as a
      hint to short-circuit the remainder of the comprehension.
    loopStep: An expression which can contain iter_var and accu_var. Computes
      the next value of accu_var.
    result: An expression which can contain accu_var. Computes the result.
  """
    accuInit = _messages.MessageField('GoogleApiExprExpr', 1)
    accuVar = _messages.StringField(2)
    iterRange = _messages.MessageField('GoogleApiExprExpr', 3)
    iterVar = _messages.StringField(4)
    loopCondition = _messages.MessageField('GoogleApiExprExpr', 5)
    loopStep = _messages.MessageField('GoogleApiExprExpr', 6)
    result = _messages.MessageField('GoogleApiExprExpr', 7)