import ast
import operator
import pyparsing
def _range_in(x, *y):
    x = ast.literal_eval(x)
    if len(y) != 4:
        raise TypeError('<range-in> operator has to be followed by 2 space separated numeric value surrounded by brackets "range_in [ 10 20 ] "')
    num_x = float(x)
    num_y = float(y[1])
    num_z = float(y[2])
    if num_y > num_z:
        raise TypeError('<range-in> operator\'s first argument has to be smaller or equal to the second argument EG"range_in  ( 10 20 ] "')
    if y[0] == '[':
        lower = num_x >= num_y
    elif y[0] == '(':
        lower = num_x > num_y
    else:
        raise TypeError('The first element should be an opening bracket ("(" or "[")')
    if y[3] == ']':
        upper = num_x <= num_z
    elif y[3] == ')':
        upper = num_x < num_z
    else:
        raise TypeError('The last element should be a closing bracket (")" or "]")')
    return lower and upper