from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
General purpose base class for AST transformations.
    
    Every visitor method can be overridden to return an AST node that has been
    altered or replaced in some way.
    