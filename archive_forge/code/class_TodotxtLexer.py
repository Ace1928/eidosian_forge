import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
class TodotxtLexer(RegexLexer):
    """
    Lexer for `Todo.txt <http://todotxt.com/>`_ todo list format.

    .. versionadded:: 2.0
    """
    name = 'Todotxt'
    aliases = ['todotxt']
    filenames = ['todo.txt', '*.todotxt']
    mimetypes = ['text/x-todo']
    CompleteTaskText = Operator
    IncompleteTaskText = Text
    Priority = Generic.Heading
    Date = Generic.Subheading
    Project = Generic.Error
    Context = String
    date_regex = '\\d{4,}-\\d{2}-\\d{2}'
    priority_regex = '\\([A-Z]\\)'
    project_regex = '\\+\\S+'
    context_regex = '@\\S+'
    complete_one_date_regex = '(x )(' + date_regex + ')'
    complete_two_date_regex = complete_one_date_regex + '( )(' + date_regex + ')'
    priority_date_regex = '(' + priority_regex + ')( )(' + date_regex + ')'
    tokens = {'root': [(complete_two_date_regex, bygroups(CompleteTaskText, Date, CompleteTaskText, Date), 'complete'), (complete_one_date_regex, bygroups(CompleteTaskText, Date), 'complete'), (priority_date_regex, bygroups(Priority, IncompleteTaskText, Date), 'incomplete'), (priority_regex, Priority, 'incomplete'), (date_regex, Date, 'incomplete'), (context_regex, Context, 'incomplete'), (project_regex, Project, 'incomplete'), ('\\S+', IncompleteTaskText, 'incomplete')], 'complete': [('\\s*\\n', CompleteTaskText, '#pop'), (context_regex, Context), (project_regex, Project), ('\\S+', CompleteTaskText), ('\\s+', CompleteTaskText)], 'incomplete': [('\\s*\\n', IncompleteTaskText, '#pop'), (context_regex, Context), (project_regex, Project), ('\\S+', IncompleteTaskText), ('\\s+', IncompleteTaskText)]}