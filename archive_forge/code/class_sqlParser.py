from antlr4 import *
from io import StringIO
import sys
class sqlParser(Parser):
    grammarFileName = 'sql.g4'
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
    sharedContextCache = PredictionContextCache()
    literalNames = ['<INVALID>', "';'", "'('", "')'", "','", "'.'", "'/*+'", "'*/'", "'->'", "'['", "']'", "':'", "'ADD'", "'AFTER'", "'ALL'", "'ALTER'", "'ANALYZE'", "'AND'", "'ANTI'", "'ANY'", "'ARCHIVE'", "'ARRAY'", "'AS'", "'ASC'", "'AT'", "'AUTHORIZATION'", "'BETWEEN'", "'BOTH'", "'BUCKET'", "'BUCKETS'", "'BY'", "'CACHE'", "'CASCADE'", "'CASE'", "'CAST'", "'CHANGE'", "'CHECK'", "'CLEAR'", "'CLUSTER'", "'CLUSTERED'", "'CODEGEN'", "'COLLATE'", "'COLLECTION'", "'COLUMN'", "'COLUMNS'", "'COMMENT'", "'COMMIT'", "'COMPACT'", "'COMPACTIONS'", "'COMPUTE'", "'CONCATENATE'", "'CONSTRAINT'", "'COST'", "'CREATE'", "'CROSS'", "'CUBE'", "'CURRENT'", "'CURRENT_DATE'", "'CURRENT_TIME'", "'CURRENT_TIMESTAMP'", "'CURRENT_USER'", "'DATA'", "'DATABASE'", '<INVALID>', "'DAY'", "'DBPROPERTIES'", "'DEFINED'", "'DELETE'", "'DELIMITED'", "'DESC'", "'DESCRIBE'", "'DFS'", "'DIRECTORIES'", "'DIRECTORY'", "'DISTINCT'", "'DISTRIBUTE'", "'DROP'", "'ELSE'", "'END'", "'ESCAPE'", "'ESCAPED'", "'EXCEPT'", "'EXCHANGE'", "'EXISTS'", "'EXPLAIN'", "'EXPORT'", "'EXTENDED'", "'EXTERNAL'", "'EXTRACT'", "'FALSE'", "'FETCH'", "'FIELDS'", "'FILTER'", "'FILEFORMAT'", "'FIRST'", "'FOLLOWING'", "'FOR'", "'FOREIGN'", "'FORMAT'", "'FORMATTED'", "'FROM'", "'FULL'", "'FUNCTION'", "'FUNCTIONS'", "'GLOBAL'", "'GRANT'", "'GROUP'", "'GROUPING'", "'HAVING'", "'HOUR'", "'IF'", "'IGNORE'", "'IMPORT'", "'IN'", "'INDEX'", "'INDEXES'", "'INNER'", "'INPATH'", "'INPUTFORMAT'", "'INSERT'", "'INTERSECT'", "'INTERVAL'", "'INTO'", "'IS'", "'ITEMS'", "'JOIN'", "'KEYS'", "'LAST'", "'LATERAL'", "'LAZY'", "'LEADING'", "'LEFT'", "'LIKE'", "'LIMIT'", "'LINES'", "'LIST'", "'LOAD'", "'LOCAL'", "'LOCATION'", "'LOCK'", "'LOCKS'", "'LOGICAL'", "'MACRO'", "'MAP'", "'MATCHED'", "'MERGE'", "'MINUTE'", "'MONTH'", "'MSCK'", "'NAMESPACE'", "'NAMESPACES'", "'NATURAL'", "'NO'", '<INVALID>', "'NULL'", "'NULLS'", "'OF'", "'ON'", "'ONLY'", "'OPTION'", "'OPTIONS'", "'OR'", "'ORDER'", "'OUT'", "'OUTER'", "'OUTPUTFORMAT'", "'OVER'", "'OVERLAPS'", "'OVERLAY'", "'OVERWRITE'", "'PARTITION'", "'PARTITIONED'", "'PARTITIONS'", "'PERCENT'", "'PIVOT'", "'PLACING'", "'POSITION'", "'PRECEDING'", "'PRIMARY'", "'PRINCIPALS'", "'PROPERTIES'", "'PURGE'", "'QUERY'", "'RANGE'", "'RECORDREADER'", "'RECORDWRITER'", "'RECOVER'", "'REDUCE'", "'REFERENCES'", "'REFRESH'", "'RENAME'", "'REPAIR'", "'REPLACE'", "'RESET'", "'RESTRICT'", "'REVOKE'", "'RIGHT'", '<INVALID>', "'ROLE'", "'ROLES'", "'ROLLBACK'", "'ROLLUP'", "'ROW'", "'ROWS'", "'SCHEMA'", "'SECOND'", "'SELECT'", "'SEMI'", "'SEPARATED'", "'SERDE'", "'SERDEPROPERTIES'", "'SESSION_USER'", "'SET'", "'MINUS'", "'SETS'", "'SHOW'", "'SKEWED'", "'SOME'", "'SORT'", "'SORTED'", "'START'", "'STATISTICS'", "'STORED'", "'STRATIFY'", "'STRUCT'", "'SUBSTR'", "'SUBSTRING'", "'TABLE'", "'TABLES'", "'TABLESAMPLE'", "'TBLPROPERTIES'", '<INVALID>', "'TERMINATED'", "'THEN'", "'TO'", "'TOUCH'", "'TRAILING'", "'TRANSACTION'", "'TRANSACTIONS'", "'TRANSFORM'", "'TRIM'", "'TRUE'", "'TRUNCATE'", "'TYPE'", "'UNARCHIVE'", "'UNBOUNDED'", "'UNCACHE'", "'UNION'", "'UNIQUE'", "'UNKNOWN'", "'UNLOCK'", "'UNSET'", "'UPDATE'", "'USE'", "'USER'", "'USING'", "'VALUES'", "'VIEW'", "'VIEWS'", "'WHEN'", "'WHERE'", "'WINDOW'", "'WITH'", "'YEAR'", '<INVALID>', "'<=>'", "'<>'", "'!='", "'<'", '<INVALID>', "'>'", '<INVALID>', "'+'", "'-'", "'*'", "'/'", "'%'", "'DIV'", "'~'", "'&'", "'|'", "'||'", "'^'"]
    symbolicNames = ['<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', 'ADD', 'AFTER', 'ALL', 'ALTER', 'ANALYZE', 'AND', 'ANTI', 'ANY', 'ARCHIVE', 'ARRAY', 'AS', 'ASC', 'AT', 'AUTHORIZATION', 'BETWEEN', 'BOTH', 'BUCKET', 'BUCKETS', 'BY', 'CACHE', 'CASCADE', 'CASE', 'CAST', 'CHANGE', 'CHECK', 'CLEAR', 'CLUSTER', 'CLUSTERED', 'CODEGEN', 'COLLATE', 'COLLECTION', 'COLUMN', 'COLUMNS', 'COMMENT', 'COMMIT', 'COMPACT', 'COMPACTIONS', 'COMPUTE', 'CONCATENATE', 'CONSTRAINT', 'COST', 'CREATE', 'CROSS', 'CUBE', 'CURRENT', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER', 'DATA', 'DATABASE', 'DATABASES', 'DAY', 'DBPROPERTIES', 'DEFINED', 'DELETE', 'DELIMITED', 'DESC', 'DESCRIBE', 'DFS', 'DIRECTORIES', 'DIRECTORY', 'DISTINCT', 'DISTRIBUTE', 'DROP', 'ELSE', 'END', 'ESCAPE', 'ESCAPED', 'EXCEPT', 'EXCHANGE', 'EXISTS', 'EXPLAIN', 'EXPORT', 'EXTENDED', 'EXTERNAL', 'EXTRACT', 'FALSE', 'FETCH', 'FIELDS', 'FILTER', 'FILEFORMAT', 'FIRST', 'FOLLOWING', 'FOR', 'FOREIGN', 'FORMAT', 'FORMATTED', 'FROM', 'FULL', 'FUNCTION', 'FUNCTIONS', 'GLOBAL', 'GRANT', 'GROUP', 'GROUPING', 'HAVING', 'HOUR', 'IF', 'IGNORE', 'IMPORT', 'IN', 'INDEX', 'INDEXES', 'INNER', 'INPATH', 'INPUTFORMAT', 'INSERT', 'INTERSECT', 'INTERVAL', 'INTO', 'IS', 'ITEMS', 'JOIN', 'KEYS', 'LAST', 'LATERAL', 'LAZY', 'LEADING', 'LEFT', 'LIKE', 'LIMIT', 'LINES', 'LIST', 'LOAD', 'LOCAL', 'LOCATION', 'LOCK', 'LOCKS', 'LOGICAL', 'MACRO', 'MAP', 'MATCHED', 'MERGE', 'MINUTE', 'MONTH', 'MSCK', 'NAMESPACE', 'NAMESPACES', 'NATURAL', 'NO', 'NOT', 'NULL', 'NULLS', 'OF', 'ON', 'ONLY', 'OPTION', 'OPTIONS', 'OR', 'ORDER', 'OUT', 'OUTER', 'OUTPUTFORMAT', 'OVER', 'OVERLAPS', 'OVERLAY', 'OVERWRITE', 'PARTITION', 'PARTITIONED', 'PARTITIONS', 'PERCENTLIT', 'PIVOT', 'PLACING', 'POSITION', 'PRECEDING', 'PRIMARY', 'PRINCIPALS', 'PROPERTIES', 'PURGE', 'QUERY', 'RANGE', 'RECORDREADER', 'RECORDWRITER', 'RECOVER', 'REDUCE', 'REFERENCES', 'REFRESH', 'RENAME', 'REPAIR', 'REPLACE', 'RESET', 'RESTRICT', 'REVOKE', 'RIGHT', 'RLIKE', 'ROLE', 'ROLES', 'ROLLBACK', 'ROLLUP', 'ROW', 'ROWS', 'SCHEMA', 'SECOND', 'SELECT', 'SEMI', 'SEPARATED', 'SERDE', 'SERDEPROPERTIES', 'SESSION_USER', 'SET', 'SETMINUS', 'SETS', 'SHOW', 'SKEWED', 'SOME', 'SORT', 'SORTED', 'START', 'STATISTICS', 'STORED', 'STRATIFY', 'STRUCT', 'SUBSTR', 'SUBSTRING', 'TABLE', 'TABLES', 'TABLESAMPLE', 'TBLPROPERTIES', 'TEMPORARY', 'TERMINATED', 'THEN', 'TO', 'TOUCH', 'TRAILING', 'TRANSACTION', 'TRANSACTIONS', 'TRANSFORM', 'TRIM', 'TRUE', 'TRUNCATE', 'TYPE', 'UNARCHIVE', 'UNBOUNDED', 'UNCACHE', 'UNION', 'UNIQUE', 'UNKNOWN', 'UNLOCK', 'UNSET', 'UPDATE', 'USE', 'USER', 'USING', 'VALUES', 'VIEW', 'VIEWS', 'WHEN', 'WHERE', 'WINDOW', 'WITH', 'YEAR', 'EQ', 'NSEQ', 'NEQ', 'NEQJ', 'LT', 'LTE', 'GT', 'GTE', 'PLUS', 'MINUS', 'ASTERISK', 'SLASH', 'PERCENT', 'DIV', 'TILDE', 'AMPERSAND', 'PIPE', 'CONCAT_PIPE', 'HAT', 'STRING', 'BIGINT_LITERAL', 'SMALLINT_LITERAL', 'TINYINT_LITERAL', 'INTEGER_VALUE', 'EXPONENT_VALUE', 'DECIMAL_VALUE', 'DOUBLE_LITERAL', 'BIGDECIMAL_LITERAL', 'IDENTIFIER', 'BACKQUOTED_IDENTIFIER', 'SIMPLE_COMMENT', 'BRACKETED_COMMENT', 'WS', 'UNRECOGNIZED']
    RULE_singleStatement = 0
    RULE_singleExpression = 1
    RULE_singleTableIdentifier = 2
    RULE_singleMultipartIdentifier = 3
    RULE_singleFunctionIdentifier = 4
    RULE_singleDataType = 5
    RULE_singleTableSchema = 6
    RULE_statement = 7
    RULE_unsupportedHiveNativeCommands = 8
    RULE_createTableHeader = 9
    RULE_replaceTableHeader = 10
    RULE_bucketSpec = 11
    RULE_skewSpec = 12
    RULE_locationSpec = 13
    RULE_commentSpec = 14
    RULE_query = 15
    RULE_insertInto = 16
    RULE_partitionSpecLocation = 17
    RULE_partitionSpec = 18
    RULE_partitionVal = 19
    RULE_namespace = 20
    RULE_describeFuncName = 21
    RULE_describeColName = 22
    RULE_ctes = 23
    RULE_namedQuery = 24
    RULE_tableProvider = 25
    RULE_createTableClauses = 26
    RULE_tablePropertyList = 27
    RULE_tableProperty = 28
    RULE_tablePropertyKey = 29
    RULE_tablePropertyValue = 30
    RULE_constantList = 31
    RULE_nestedConstantList = 32
    RULE_createFileFormat = 33
    RULE_fileFormat = 34
    RULE_storageHandler = 35
    RULE_resource = 36
    RULE_dmlStatementNoWith = 37
    RULE_queryOrganization = 38
    RULE_multiInsertQueryBody = 39
    RULE_queryTerm = 40
    RULE_queryPrimary = 41
    RULE_sortItem = 42
    RULE_fromStatement = 43
    RULE_fromStatementBody = 44
    RULE_querySpecification = 45
    RULE_transformClause = 46
    RULE_selectClause = 47
    RULE_setClause = 48
    RULE_matchedClause = 49
    RULE_notMatchedClause = 50
    RULE_matchedAction = 51
    RULE_notMatchedAction = 52
    RULE_assignmentList = 53
    RULE_assignment = 54
    RULE_whereClause = 55
    RULE_havingClause = 56
    RULE_hint = 57
    RULE_hintStatement = 58
    RULE_fromClause = 59
    RULE_aggregationClause = 60
    RULE_groupingSet = 61
    RULE_pivotClause = 62
    RULE_pivotColumn = 63
    RULE_pivotValue = 64
    RULE_lateralView = 65
    RULE_setQuantifier = 66
    RULE_relation = 67
    RULE_joinRelation = 68
    RULE_joinType = 69
    RULE_joinCriteria = 70
    RULE_sample = 71
    RULE_sampleMethod = 72
    RULE_identifierList = 73
    RULE_identifierSeq = 74
    RULE_orderedIdentifierList = 75
    RULE_orderedIdentifier = 76
    RULE_identifierCommentList = 77
    RULE_identifierComment = 78
    RULE_relationPrimary = 79
    RULE_inlineTable = 80
    RULE_functionTable = 81
    RULE_tableAlias = 82
    RULE_rowFormat = 83
    RULE_multipartIdentifierList = 84
    RULE_multipartIdentifier = 85
    RULE_tableIdentifier = 86
    RULE_functionIdentifier = 87
    RULE_namedExpression = 88
    RULE_namedExpressionSeq = 89
    RULE_transformList = 90
    RULE_transform = 91
    RULE_transformArgument = 92
    RULE_expression = 93
    RULE_booleanExpression = 94
    RULE_predicate = 95
    RULE_valueExpression = 96
    RULE_primaryExpression = 97
    RULE_constant = 98
    RULE_comparisonOperator = 99
    RULE_arithmeticOperator = 100
    RULE_predicateOperator = 101
    RULE_booleanValue = 102
    RULE_interval = 103
    RULE_errorCapturingMultiUnitsInterval = 104
    RULE_multiUnitsInterval = 105
    RULE_errorCapturingUnitToUnitInterval = 106
    RULE_unitToUnitInterval = 107
    RULE_intervalValue = 108
    RULE_intervalUnit = 109
    RULE_colPosition = 110
    RULE_dataType = 111
    RULE_qualifiedColTypeWithPositionList = 112
    RULE_qualifiedColTypeWithPosition = 113
    RULE_colTypeList = 114
    RULE_colType = 115
    RULE_complexColTypeList = 116
    RULE_complexColType = 117
    RULE_whenClause = 118
    RULE_windowClause = 119
    RULE_namedWindow = 120
    RULE_windowSpec = 121
    RULE_windowFrame = 122
    RULE_frameBound = 123
    RULE_qualifiedNameList = 124
    RULE_functionName = 125
    RULE_qualifiedName = 126
    RULE_errorCapturingIdentifier = 127
    RULE_errorCapturingIdentifierExtra = 128
    RULE_identifier = 129
    RULE_strictIdentifier = 130
    RULE_quotedIdentifier = 131
    RULE_number = 132
    RULE_alterColumnAction = 133
    RULE_ansiNonReserved = 134
    RULE_strictNonReserved = 135
    RULE_nonReserved = 136
    ruleNames = ['singleStatement', 'singleExpression', 'singleTableIdentifier', 'singleMultipartIdentifier', 'singleFunctionIdentifier', 'singleDataType', 'singleTableSchema', 'statement', 'unsupportedHiveNativeCommands', 'createTableHeader', 'replaceTableHeader', 'bucketSpec', 'skewSpec', 'locationSpec', 'commentSpec', 'query', 'insertInto', 'partitionSpecLocation', 'partitionSpec', 'partitionVal', 'namespace', 'describeFuncName', 'describeColName', 'ctes', 'namedQuery', 'tableProvider', 'createTableClauses', 'tablePropertyList', 'tableProperty', 'tablePropertyKey', 'tablePropertyValue', 'constantList', 'nestedConstantList', 'createFileFormat', 'fileFormat', 'storageHandler', 'resource', 'dmlStatementNoWith', 'queryOrganization', 'multiInsertQueryBody', 'queryTerm', 'queryPrimary', 'sortItem', 'fromStatement', 'fromStatementBody', 'querySpecification', 'transformClause', 'selectClause', 'setClause', 'matchedClause', 'notMatchedClause', 'matchedAction', 'notMatchedAction', 'assignmentList', 'assignment', 'whereClause', 'havingClause', 'hint', 'hintStatement', 'fromClause', 'aggregationClause', 'groupingSet', 'pivotClause', 'pivotColumn', 'pivotValue', 'lateralView', 'setQuantifier', 'relation', 'joinRelation', 'joinType', 'joinCriteria', 'sample', 'sampleMethod', 'identifierList', 'identifierSeq', 'orderedIdentifierList', 'orderedIdentifier', 'identifierCommentList', 'identifierComment', 'relationPrimary', 'inlineTable', 'functionTable', 'tableAlias', 'rowFormat', 'multipartIdentifierList', 'multipartIdentifier', 'tableIdentifier', 'functionIdentifier', 'namedExpression', 'namedExpressionSeq', 'transformList', 'transform', 'transformArgument', 'expression', 'booleanExpression', 'predicate', 'valueExpression', 'primaryExpression', 'constant', 'comparisonOperator', 'arithmeticOperator', 'predicateOperator', 'booleanValue', 'interval', 'errorCapturingMultiUnitsInterval', 'multiUnitsInterval', 'errorCapturingUnitToUnitInterval', 'unitToUnitInterval', 'intervalValue', 'intervalUnit', 'colPosition', 'dataType', 'qualifiedColTypeWithPositionList', 'qualifiedColTypeWithPosition', 'colTypeList', 'colType', 'complexColTypeList', 'complexColType', 'whenClause', 'windowClause', 'namedWindow', 'windowSpec', 'windowFrame', 'frameBound', 'qualifiedNameList', 'functionName', 'qualifiedName', 'errorCapturingIdentifier', 'errorCapturingIdentifierExtra', 'identifier', 'strictIdentifier', 'quotedIdentifier', 'number', 'alterColumnAction', 'ansiNonReserved', 'strictNonReserved', 'nonReserved']
    EOF = Token.EOF
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    ADD = 12
    AFTER = 13
    ALL = 14
    ALTER = 15
    ANALYZE = 16
    AND = 17
    ANTI = 18
    ANY = 19
    ARCHIVE = 20
    ARRAY = 21
    AS = 22
    ASC = 23
    AT = 24
    AUTHORIZATION = 25
    BETWEEN = 26
    BOTH = 27
    BUCKET = 28
    BUCKETS = 29
    BY = 30
    CACHE = 31
    CASCADE = 32
    CASE = 33
    CAST = 34
    CHANGE = 35
    CHECK = 36
    CLEAR = 37
    CLUSTER = 38
    CLUSTERED = 39
    CODEGEN = 40
    COLLATE = 41
    COLLECTION = 42
    COLUMN = 43
    COLUMNS = 44
    COMMENT = 45
    COMMIT = 46
    COMPACT = 47
    COMPACTIONS = 48
    COMPUTE = 49
    CONCATENATE = 50
    CONSTRAINT = 51
    COST = 52
    CREATE = 53
    CROSS = 54
    CUBE = 55
    CURRENT = 56
    CURRENT_DATE = 57
    CURRENT_TIME = 58
    CURRENT_TIMESTAMP = 59
    CURRENT_USER = 60
    DATA = 61
    DATABASE = 62
    DATABASES = 63
    DAY = 64
    DBPROPERTIES = 65
    DEFINED = 66
    DELETE = 67
    DELIMITED = 68
    DESC = 69
    DESCRIBE = 70
    DFS = 71
    DIRECTORIES = 72
    DIRECTORY = 73
    DISTINCT = 74
    DISTRIBUTE = 75
    DROP = 76
    ELSE = 77
    END = 78
    ESCAPE = 79
    ESCAPED = 80
    EXCEPT = 81
    EXCHANGE = 82
    EXISTS = 83
    EXPLAIN = 84
    EXPORT = 85
    EXTENDED = 86
    EXTERNAL = 87
    EXTRACT = 88
    FALSE = 89
    FETCH = 90
    FIELDS = 91
    FILTER = 92
    FILEFORMAT = 93
    FIRST = 94
    FOLLOWING = 95
    FOR = 96
    FOREIGN = 97
    FORMAT = 98
    FORMATTED = 99
    FROM = 100
    FULL = 101
    FUNCTION = 102
    FUNCTIONS = 103
    GLOBAL = 104
    GRANT = 105
    GROUP = 106
    GROUPING = 107
    HAVING = 108
    HOUR = 109
    IF = 110
    IGNORE = 111
    IMPORT = 112
    IN = 113
    INDEX = 114
    INDEXES = 115
    INNER = 116
    INPATH = 117
    INPUTFORMAT = 118
    INSERT = 119
    INTERSECT = 120
    INTERVAL = 121
    INTO = 122
    IS = 123
    ITEMS = 124
    JOIN = 125
    KEYS = 126
    LAST = 127
    LATERAL = 128
    LAZY = 129
    LEADING = 130
    LEFT = 131
    LIKE = 132
    LIMIT = 133
    LINES = 134
    LIST = 135
    LOAD = 136
    LOCAL = 137
    LOCATION = 138
    LOCK = 139
    LOCKS = 140
    LOGICAL = 141
    MACRO = 142
    MAP = 143
    MATCHED = 144
    MERGE = 145
    MINUTE = 146
    MONTH = 147
    MSCK = 148
    NAMESPACE = 149
    NAMESPACES = 150
    NATURAL = 151
    NO = 152
    NOT = 153
    NULL = 154
    NULLS = 155
    OF = 156
    ON = 157
    ONLY = 158
    OPTION = 159
    OPTIONS = 160
    OR = 161
    ORDER = 162
    OUT = 163
    OUTER = 164
    OUTPUTFORMAT = 165
    OVER = 166
    OVERLAPS = 167
    OVERLAY = 168
    OVERWRITE = 169
    PARTITION = 170
    PARTITIONED = 171
    PARTITIONS = 172
    PERCENTLIT = 173
    PIVOT = 174
    PLACING = 175
    POSITION = 176
    PRECEDING = 177
    PRIMARY = 178
    PRINCIPALS = 179
    PROPERTIES = 180
    PURGE = 181
    QUERY = 182
    RANGE = 183
    RECORDREADER = 184
    RECORDWRITER = 185
    RECOVER = 186
    REDUCE = 187
    REFERENCES = 188
    REFRESH = 189
    RENAME = 190
    REPAIR = 191
    REPLACE = 192
    RESET = 193
    RESTRICT = 194
    REVOKE = 195
    RIGHT = 196
    RLIKE = 197
    ROLE = 198
    ROLES = 199
    ROLLBACK = 200
    ROLLUP = 201
    ROW = 202
    ROWS = 203
    SCHEMA = 204
    SECOND = 205
    SELECT = 206
    SEMI = 207
    SEPARATED = 208
    SERDE = 209
    SERDEPROPERTIES = 210
    SESSION_USER = 211
    SET = 212
    SETMINUS = 213
    SETS = 214
    SHOW = 215
    SKEWED = 216
    SOME = 217
    SORT = 218
    SORTED = 219
    START = 220
    STATISTICS = 221
    STORED = 222
    STRATIFY = 223
    STRUCT = 224
    SUBSTR = 225
    SUBSTRING = 226
    TABLE = 227
    TABLES = 228
    TABLESAMPLE = 229
    TBLPROPERTIES = 230
    TEMPORARY = 231
    TERMINATED = 232
    THEN = 233
    TO = 234
    TOUCH = 235
    TRAILING = 236
    TRANSACTION = 237
    TRANSACTIONS = 238
    TRANSFORM = 239
    TRIM = 240
    TRUE = 241
    TRUNCATE = 242
    TYPE = 243
    UNARCHIVE = 244
    UNBOUNDED = 245
    UNCACHE = 246
    UNION = 247
    UNIQUE = 248
    UNKNOWN = 249
    UNLOCK = 250
    UNSET = 251
    UPDATE = 252
    USE = 253
    USER = 254
    USING = 255
    VALUES = 256
    VIEW = 257
    VIEWS = 258
    WHEN = 259
    WHERE = 260
    WINDOW = 261
    WITH = 262
    YEAR = 263
    EQ = 264
    NSEQ = 265
    NEQ = 266
    NEQJ = 267
    LT = 268
    LTE = 269
    GT = 270
    GTE = 271
    PLUS = 272
    MINUS = 273
    ASTERISK = 274
    SLASH = 275
    PERCENT = 276
    DIV = 277
    TILDE = 278
    AMPERSAND = 279
    PIPE = 280
    CONCAT_PIPE = 281
    HAT = 282
    STRING = 283
    BIGINT_LITERAL = 284
    SMALLINT_LITERAL = 285
    TINYINT_LITERAL = 286
    INTEGER_VALUE = 287
    EXPONENT_VALUE = 288
    DECIMAL_VALUE = 289
    DOUBLE_LITERAL = 290
    BIGDECIMAL_LITERAL = 291
    IDENTIFIER = 292
    BACKQUOTED_IDENTIFIER = 293
    SIMPLE_COMMENT = 294
    BRACKETED_COMMENT = 295
    WS = 296
    UNRECOGNIZED = 297

    def __init__(self, input: TokenStream, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.11.1')
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None

    @property
    def legacy_setops_precedence_enbled(self):
        return False

    @property
    def legacy_exponent_literal_as_decimal_enabled(self):
        return False

    @property
    def isValidDecimal(self):
        return True
        nextChar = self._input.LA(1)
        if nextChar >= 'A' and nextChar <= 'Z' or (nextChar >= '0' and nextChar <= '9') or nextChar == '_':
            return False
        else:
            return True

    @property
    def SQL_standard_keyword_behavior(self):
        if '_ansi_sql' in self.__dict__:
            return self._ansi_sql
        return False

    def isHint(self):
        return False
        nextChar = self._input.LA(1)
        if nextChar == '+':
            return True
        else:
            return False

    @property
    def allUpperCase(self):
        if '_all_upper_case' in self.__dict__:
            return self._all_upper_case
        return False

    class SingleStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statement(self):
            return self.getTypedRuleContext(sqlParser.StatementContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleStatement'):
                return visitor.visitSingleStatement(self)
            else:
                return visitor.visitChildren(self)

    def singleStatement(self):
        localctx = sqlParser.SingleStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_singleStatement)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 274
            self.statement()
            self.state = 278
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 1:
                self.state = 275
                self.match(sqlParser.T__0)
                self.state = 280
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 281
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def namedExpression(self):
            return self.getTypedRuleContext(sqlParser.NamedExpressionContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleExpression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleExpression'):
                return visitor.visitSingleExpression(self)
            else:
                return visitor.visitChildren(self)

    def singleExpression(self):
        localctx = sqlParser.SingleExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_singleExpression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 283
            self.namedExpression()
            self.state = 284
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleTableIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def tableIdentifier(self):
            return self.getTypedRuleContext(sqlParser.TableIdentifierContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleTableIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleTableIdentifier'):
                return visitor.visitSingleTableIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleTableIdentifier(self):
        localctx = sqlParser.SingleTableIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_singleTableIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 286
            self.tableIdentifier()
            self.state = 287
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleMultipartIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleMultipartIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleMultipartIdentifier'):
                return visitor.visitSingleMultipartIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleMultipartIdentifier(self):
        localctx = sqlParser.SingleMultipartIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_singleMultipartIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 289
            self.multipartIdentifier()
            self.state = 290
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleFunctionIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def functionIdentifier(self):
            return self.getTypedRuleContext(sqlParser.FunctionIdentifierContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleFunctionIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleFunctionIdentifier'):
                return visitor.visitSingleFunctionIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleFunctionIdentifier(self):
        localctx = sqlParser.SingleFunctionIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_singleFunctionIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 292
            self.functionIdentifier()
            self.state = 293
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleDataTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleDataType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleDataType'):
                return visitor.visitSingleDataType(self)
            else:
                return visitor.visitChildren(self)

    def singleDataType(self):
        localctx = sqlParser.SingleDataTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_singleDataType)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 295
            self.dataType()
            self.state = 296
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleTableSchemaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def colTypeList(self):
            return self.getTypedRuleContext(sqlParser.ColTypeListContext, 0)

        def EOF(self):
            return self.getToken(sqlParser.EOF, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_singleTableSchema

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleTableSchema'):
                return visitor.visitSingleTableSchema(self)
            else:
                return visitor.visitChildren(self)

    def singleTableSchema(self):
        localctx = sqlParser.SingleTableSchemaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_singleTableSchema)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 298
            self.colTypeList()
            self.state = 299
            self.match(sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_statement

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ExplainContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXPLAIN(self):
            return self.getToken(sqlParser.EXPLAIN, 0)

        def statement(self):
            return self.getTypedRuleContext(sqlParser.StatementContext, 0)

        def LOGICAL(self):
            return self.getToken(sqlParser.LOGICAL, 0)

        def FORMATTED(self):
            return self.getToken(sqlParser.FORMATTED, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def CODEGEN(self):
            return self.getToken(sqlParser.CODEGEN, 0)

        def COST(self):
            return self.getToken(sqlParser.COST, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExplain'):
                return visitor.visitExplain(self)
            else:
                return visitor.visitChildren(self)

    class ResetConfigurationContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def RESET(self):
            return self.getToken(sqlParser.RESET, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitResetConfiguration'):
                return visitor.visitResetConfiguration(self)
            else:
                return visitor.visitChildren(self)

    class AlterViewQueryContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAlterViewQuery'):
                return visitor.visitAlterViewQuery(self)
            else:
                return visitor.visitChildren(self)

    class UseContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def USE(self):
            return self.getToken(sqlParser.USE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def NAMESPACE(self):
            return self.getToken(sqlParser.NAMESPACE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUse'):
                return visitor.visitUse(self)
            else:
                return visitor.visitChildren(self)

    class DropNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def RESTRICT(self):
            return self.getToken(sqlParser.RESTRICT, 0)

        def CASCADE(self):
            return self.getToken(sqlParser.CASCADE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropNamespace'):
                return visitor.visitDropNamespace(self)
            else:
                return visitor.visitChildren(self)

    class CreateTempViewUsingContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def tableIdentifier(self):
            return self.getTypedRuleContext(sqlParser.TableIdentifierContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(sqlParser.TableProviderContext, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def GLOBAL(self):
            return self.getToken(sqlParser.GLOBAL, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(sqlParser.ColTypeListContext, 0)

        def OPTIONS(self):
            return self.getToken(sqlParser.OPTIONS, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTempViewUsing'):
                return visitor.visitCreateTempViewUsing(self)
            else:
                return visitor.visitChildren(self)

    class RenameTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.qpdfrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def RENAME(self):
            return self.getToken(sqlParser.RENAME, 0)

        def TO(self):
            return self.getToken(sqlParser.TO, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRenameTable'):
                return visitor.visitRenameTable(self)
            else:
                return visitor.visitChildren(self)

    class FailNativeCommandContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def ROLE(self):
            return self.getToken(sqlParser.ROLE, 0)

        def unsupportedHiveNativeCommands(self):
            return self.getTypedRuleContext(sqlParser.UnsupportedHiveNativeCommandsContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFailNativeCommand'):
                return visitor.visitFailNativeCommand(self)
            else:
                return visitor.visitChildren(self)

    class ClearCacheContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def CLEAR(self):
            return self.getToken(sqlParser.CLEAR, 0)

        def CACHE(self):
            return self.getToken(sqlParser.CACHE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitClearCache'):
                return visitor.visitClearCache(self)
            else:
                return visitor.visitChildren(self)

    class DropViewContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropView'):
                return visitor.visitDropView(self)
            else:
                return visitor.visitChildren(self)

    class ShowTablesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.pattern = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def TABLES(self):
            return self.getToken(sqlParser.TABLES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowTables'):
                return visitor.visitShowTables(self)
            else:
                return visitor.visitChildren(self)

    class RecoverPartitionsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def RECOVER(self):
            return self.getToken(sqlParser.RECOVER, 0)

        def PARTITIONS(self):
            return self.getToken(sqlParser.PARTITIONS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRecoverPartitions'):
                return visitor.visitRecoverPartitions(self)
            else:
                return visitor.visitChildren(self)

    class ShowCurrentNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def CURRENT(self):
            return self.getToken(sqlParser.CURRENT, 0)

        def NAMESPACE(self):
            return self.getToken(sqlParser.NAMESPACE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowCurrentNamespace'):
                return visitor.visitShowCurrentNamespace(self)
            else:
                return visitor.visitChildren(self)

    class RenameTablePartitionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.qpdfrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def RENAME(self):
            return self.getToken(sqlParser.RENAME, 0)

        def TO(self):
            return self.getToken(sqlParser.TO, 0)

        def partitionSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PartitionSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.PartitionSpecContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRenameTablePartition'):
                return visitor.visitRenameTablePartition(self)
            else:
                return visitor.visitChildren(self)

    class RepairTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def MSCK(self):
            return self.getToken(sqlParser.MSCK, 0)

        def REPAIR(self):
            return self.getToken(sqlParser.REPAIR, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRepairTable'):
                return visitor.visitRepairTable(self)
            else:
                return visitor.visitChildren(self)

    class RefreshResourceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def REFRESH(self):
            return self.getToken(sqlParser.REFRESH, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRefreshResource'):
                return visitor.visitRefreshResource(self)
            else:
                return visitor.visitChildren(self)

    class ShowCreateTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def SERDE(self):
            return self.getToken(sqlParser.SERDE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowCreateTable'):
                return visitor.visitShowCreateTable(self)
            else:
                return visitor.visitChildren(self)

    class ShowNamespacesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.pattern = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def DATABASES(self):
            return self.getToken(sqlParser.DATABASES, 0)

        def NAMESPACES(self):
            return self.getToken(sqlParser.NAMESPACES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowNamespaces'):
                return visitor.visitShowNamespaces(self)
            else:
                return visitor.visitChildren(self)

    class ShowColumnsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.ns = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def FROM(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.FROM)
            else:
                return self.getToken(sqlParser.FROM, i)

        def IN(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.IN)
            else:
                return self.getToken(sqlParser.IN, i)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowColumns'):
                return visitor.visitShowColumns(self)
            else:
                return visitor.visitChildren(self)

    class ReplaceTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def replaceTableHeader(self):
            return self.getTypedRuleContext(sqlParser.ReplaceTableHeaderContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(sqlParser.TableProviderContext, 0)

        def createTableClauses(self):
            return self.getTypedRuleContext(sqlParser.CreateTableClausesContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(sqlParser.ColTypeListContext, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitReplaceTable'):
                return visitor.visitReplaceTable(self)
            else:
                return visitor.visitChildren(self)

    class AddTablePartitionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def ADD(self):
            return self.getToken(sqlParser.ADD, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def partitionSpecLocation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PartitionSpecLocationContext)
            else:
                return self.getTypedRuleContext(sqlParser.PartitionSpecLocationContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAddTablePartition'):
                return visitor.visitAddTablePartition(self)
            else:
                return visitor.visitChildren(self)

    class SetNamespaceLocationContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(sqlParser.LocationSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetNamespaceLocation'):
                return visitor.visitSetNamespaceLocation(self)
            else:
                return visitor.visitChildren(self)

    class RefreshTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def REFRESH(self):
            return self.getToken(sqlParser.REFRESH, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRefreshTable'):
                return visitor.visitRefreshTable(self)
            else:
                return visitor.visitChildren(self)

    class SetNamespacePropertiesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def DBPROPERTIES(self):
            return self.getToken(sqlParser.DBPROPERTIES, 0)

        def PROPERTIES(self):
            return self.getToken(sqlParser.PROPERTIES, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetNamespaceProperties'):
                return visitor.visitSetNamespaceProperties(self)
            else:
                return visitor.visitChildren(self)

    class ManageResourceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.op = None
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def ADD(self):
            return self.getToken(sqlParser.ADD, 0)

        def LIST(self):
            return self.getToken(sqlParser.LIST, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitManageResource'):
                return visitor.visitManageResource(self)
            else:
                return visitor.visitChildren(self)

    class AnalyzeContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ANALYZE(self):
            return self.getToken(sqlParser.ANALYZE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def COMPUTE(self):
            return self.getToken(sqlParser.COMPUTE, 0)

        def STATISTICS(self):
            return self.getToken(sqlParser.STATISTICS, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def FOR(self):
            return self.getToken(sqlParser.FOR, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def identifierSeq(self):
            return self.getTypedRuleContext(sqlParser.IdentifierSeqContext, 0)

        def ALL(self):
            return self.getToken(sqlParser.ALL, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAnalyze'):
                return visitor.visitAnalyze(self)
            else:
                return visitor.visitChildren(self)

    class CreateHiveTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.columns = None
            self.partitionColumns = None
            self.partitionColumnNames = None
            self.tableProps = None
            self.copyFrom(ctx)

        def createTableHeader(self):
            return self.getTypedRuleContext(sqlParser.CreateTableHeaderContext, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.CommentSpecContext, i)

        def bucketSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.BucketSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.BucketSpecContext, i)

        def skewSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.SkewSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.SkewSpecContext, i)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(sqlParser.RowFormatContext, i)

        def createFileFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CreateFileFormatContext)
            else:
                return self.getTypedRuleContext(sqlParser.CreateFileFormatContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.LocationSpecContext, i)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def colTypeList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ColTypeListContext)
            else:
                return self.getTypedRuleContext(sqlParser.ColTypeListContext, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.PARTITIONED)
            else:
                return self.getToken(sqlParser.PARTITIONED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(sqlParser.TBLPROPERTIES, i)

        def identifierList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierListContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierListContext, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyListContext, i)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateHiveTable'):
                return visitor.visitCreateHiveTable(self)
            else:
                return visitor.visitChildren(self)

    class CreateFunctionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.className = None
            self.copyFrom(ctx)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def FUNCTION(self):
            return self.getToken(sqlParser.FUNCTION, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def resource(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ResourceContext)
            else:
                return self.getTypedRuleContext(sqlParser.ResourceContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateFunction'):
                return visitor.visitCreateFunction(self)
            else:
                return visitor.visitChildren(self)

    class ShowTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.ns = None
            self.pattern = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowTable'):
                return visitor.visitShowTable(self)
            else:
                return visitor.visitChildren(self)

    class HiveReplaceColumnsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.columns = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def qualifiedColTypeWithPositionList(self):
            return self.getTypedRuleContext(sqlParser.QualifiedColTypeWithPositionListContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHiveReplaceColumns'):
                return visitor.visitHiveReplaceColumns(self)
            else:
                return visitor.visitChildren(self)

    class CommentNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.comment = None
            self.copyFrom(ctx)

        def COMMENT(self):
            return self.getToken(sqlParser.COMMENT, 0)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IS(self):
            return self.getToken(sqlParser.IS, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCommentNamespace'):
                return visitor.visitCommentNamespace(self)
            else:
                return visitor.visitChildren(self)

    class CreateTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def createTableHeader(self):
            return self.getTypedRuleContext(sqlParser.CreateTableHeaderContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(sqlParser.TableProviderContext, 0)

        def createTableClauses(self):
            return self.getTypedRuleContext(sqlParser.CreateTableClausesContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(sqlParser.ColTypeListContext, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTable'):
                return visitor.visitCreateTable(self)
            else:
                return visitor.visitChildren(self)

    class DmlStatementContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def dmlStatementNoWith(self):
            return self.getTypedRuleContext(sqlParser.DmlStatementNoWithContext, 0)

        def ctes(self):
            return self.getTypedRuleContext(sqlParser.CtesContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDmlStatement'):
                return visitor.visitDmlStatement(self)
            else:
                return visitor.visitChildren(self)

    class CreateTableLikeContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.target = None
            self.source = None
            self.tableProps = None
            self.copyFrom(ctx)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def tableIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TableIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.TableIdentifierContext, i)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def tableProvider(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TableProviderContext)
            else:
                return self.getTypedRuleContext(sqlParser.TableProviderContext, i)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(sqlParser.RowFormatContext, i)

        def createFileFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CreateFileFormatContext)
            else:
                return self.getTypedRuleContext(sqlParser.CreateFileFormatContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.LocationSpecContext, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyListContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTableLike'):
                return visitor.visitCreateTableLike(self)
            else:
                return visitor.visitChildren(self)

    class UncacheTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def UNCACHE(self):
            return self.getToken(sqlParser.UNCACHE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUncacheTable'):
                return visitor.visitUncacheTable(self)
            else:
                return visitor.visitChildren(self)

    class DropFunctionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def FUNCTION(self):
            return self.getToken(sqlParser.FUNCTION, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropFunction'):
                return visitor.visitDropFunction(self)
            else:
                return visitor.visitChildren(self)

    class DescribeRelationContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.option = None
            self.copyFrom(ctx)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def describeColName(self):
            return self.getTypedRuleContext(sqlParser.DescribeColNameContext, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def FORMATTED(self):
            return self.getToken(sqlParser.FORMATTED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeRelation'):
                return visitor.visitDescribeRelation(self)
            else:
                return visitor.visitChildren(self)

    class LoadDataContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.path = None
            self.copyFrom(ctx)

        def LOAD(self):
            return self.getToken(sqlParser.LOAD, 0)

        def DATA(self):
            return self.getToken(sqlParser.DATA, 0)

        def INPATH(self):
            return self.getToken(sqlParser.INPATH, 0)

        def INTO(self):
            return self.getToken(sqlParser.INTO, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def LOCAL(self):
            return self.getToken(sqlParser.LOCAL, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLoadData'):
                return visitor.visitLoadData(self)
            else:
                return visitor.visitChildren(self)

    class ShowPartitionsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def PARTITIONS(self):
            return self.getToken(sqlParser.PARTITIONS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowPartitions'):
                return visitor.visitShowPartitions(self)
            else:
                return visitor.visitChildren(self)

    class DescribeFunctionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def FUNCTION(self):
            return self.getToken(sqlParser.FUNCTION, 0)

        def describeFuncName(self):
            return self.getTypedRuleContext(sqlParser.DescribeFuncNameContext, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeFunction'):
                return visitor.visitDescribeFunction(self)
            else:
                return visitor.visitChildren(self)

    class RenameTableColumnContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.qpdfrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def RENAME(self):
            return self.getToken(sqlParser.RENAME, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def TO(self):
            return self.getToken(sqlParser.TO, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRenameTableColumn'):
                return visitor.visitRenameTableColumn(self)
            else:
                return visitor.visitChildren(self)

    class StatementDefaultContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStatementDefault'):
                return visitor.visitStatementDefault(self)
            else:
                return visitor.visitChildren(self)

    class HiveChangeColumnContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.colName = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def CHANGE(self):
            return self.getToken(sqlParser.CHANGE, 0)

        def colType(self):
            return self.getTypedRuleContext(sqlParser.ColTypeContext, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def colPosition(self):
            return self.getTypedRuleContext(sqlParser.ColPositionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHiveChangeColumn'):
                return visitor.visitHiveChangeColumn(self)
            else:
                return visitor.visitChildren(self)

    class DescribeQueryContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def QUERY(self):
            return self.getToken(sqlParser.QUERY, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeQuery'):
                return visitor.visitDescribeQuery(self)
            else:
                return visitor.visitChildren(self)

    class TruncateTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def TRUNCATE(self):
            return self.getToken(sqlParser.TRUNCATE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTruncateTable'):
                return visitor.visitTruncateTable(self)
            else:
                return visitor.visitChildren(self)

    class SetTableSerDeContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def SERDE(self):
            return self.getToken(sqlParser.SERDE, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetTableSerDe'):
                return visitor.visitSetTableSerDe(self)
            else:
                return visitor.visitChildren(self)

    class CreateViewContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def identifierCommentList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierCommentListContext, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.CommentSpecContext, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.PARTITIONED)
            else:
                return self.getToken(sqlParser.PARTITIONED, i)

        def ON(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.ON)
            else:
                return self.getToken(sqlParser.ON, i)

        def identifierList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierListContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierListContext, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyListContext, i)

        def GLOBAL(self):
            return self.getToken(sqlParser.GLOBAL, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateView'):
                return visitor.visitCreateView(self)
            else:
                return visitor.visitChildren(self)

    class DropTablePartitionsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def partitionSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PartitionSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.PartitionSpecContext, i)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def PURGE(self):
            return self.getToken(sqlParser.PURGE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropTablePartitions'):
                return visitor.visitDropTablePartitions(self)
            else:
                return visitor.visitChildren(self)

    class SetConfigurationContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetConfiguration'):
                return visitor.visitSetConfiguration(self)
            else:
                return visitor.visitChildren(self)

    class DropTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def PURGE(self):
            return self.getToken(sqlParser.PURGE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropTable'):
                return visitor.visitDropTable(self)
            else:
                return visitor.visitChildren(self)

    class DescribeNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeNamespace'):
                return visitor.visitDescribeNamespace(self)
            else:
                return visitor.visitChildren(self)

    class AlterTableAlterColumnContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.column = None
            self.copyFrom(ctx)

        def ALTER(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.ALTER)
            else:
                return self.getToken(sqlParser.ALTER, i)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def CHANGE(self):
            return self.getToken(sqlParser.CHANGE, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def alterColumnAction(self):
            return self.getTypedRuleContext(sqlParser.AlterColumnActionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAlterTableAlterColumn'):
                return visitor.visitAlterTableAlterColumn(self)
            else:
                return visitor.visitChildren(self)

    class CommentTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.comment = None
            self.copyFrom(ctx)

        def COMMENT(self):
            return self.getToken(sqlParser.COMMENT, 0)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IS(self):
            return self.getToken(sqlParser.IS, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCommentTable'):
                return visitor.visitCommentTable(self)
            else:
                return visitor.visitChildren(self)

    class CreateNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def namespace(self):
            return self.getTypedRuleContext(sqlParser.NamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.CommentSpecContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.LocationSpecContext, i)

        def WITH(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.WITH)
            else:
                return self.getToken(sqlParser.WITH, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyListContext, i)

        def DBPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.DBPROPERTIES)
            else:
                return self.getToken(sqlParser.DBPROPERTIES, i)

        def PROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.PROPERTIES)
            else:
                return self.getToken(sqlParser.PROPERTIES, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateNamespace'):
                return visitor.visitCreateNamespace(self)
            else:
                return visitor.visitChildren(self)

    class ShowTblPropertiesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.key = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def TBLPROPERTIES(self):
            return self.getToken(sqlParser.TBLPROPERTIES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def tablePropertyKey(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyKeyContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowTblProperties'):
                return visitor.visitShowTblProperties(self)
            else:
                return visitor.visitChildren(self)

    class UnsetTablePropertiesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def UNSET(self):
            return self.getToken(sqlParser.UNSET, 0)

        def TBLPROPERTIES(self):
            return self.getToken(sqlParser.TBLPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnsetTableProperties'):
                return visitor.visitUnsetTableProperties(self)
            else:
                return visitor.visitChildren(self)

    class SetTableLocationContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(sqlParser.LocationSpecContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetTableLocation'):
                return visitor.visitSetTableLocation(self)
            else:
                return visitor.visitChildren(self)

    class DropTableColumnsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.columns = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def multipartIdentifierList(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropTableColumns'):
                return visitor.visitDropTableColumns(self)
            else:
                return visitor.visitChildren(self)

    class ShowViewsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.pattern = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def VIEWS(self):
            return self.getToken(sqlParser.VIEWS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowViews'):
                return visitor.visitShowViews(self)
            else:
                return visitor.visitChildren(self)

    class ShowFunctionsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.pattern = None
            self.copyFrom(ctx)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def FUNCTIONS(self):
            return self.getToken(sqlParser.FUNCTIONS, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowFunctions'):
                return visitor.visitShowFunctions(self)
            else:
                return visitor.visitChildren(self)

    class CacheTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.options = None
            self.copyFrom(ctx)

        def CACHE(self):
            return self.getToken(sqlParser.CACHE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def LAZY(self):
            return self.getToken(sqlParser.LAZY, 0)

        def OPTIONS(self):
            return self.getToken(sqlParser.OPTIONS, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCacheTable'):
                return visitor.visitCacheTable(self)
            else:
                return visitor.visitChildren(self)

    class AddTableColumnsContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.columns = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def ADD(self):
            return self.getToken(sqlParser.ADD, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def qualifiedColTypeWithPositionList(self):
            return self.getTypedRuleContext(sqlParser.QualifiedColTypeWithPositionListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAddTableColumns'):
                return visitor.visitAddTableColumns(self)
            else:
                return visitor.visitChildren(self)

    class SetTablePropertiesContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def TBLPROPERTIES(self):
            return self.getToken(sqlParser.TBLPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetTableProperties'):
                return visitor.visitSetTableProperties(self)
            else:
                return visitor.visitChildren(self)

    def statement(self):
        localctx = sqlParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_statement)
        self._la = 0
        try:
            self.state = 1006
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 106, self._ctx)
            if la_ == 1:
                localctx = sqlParser.StatementDefaultContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 301
                self.query()
                pass
            elif la_ == 2:
                localctx = sqlParser.DmlStatementContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 303
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 262:
                    self.state = 302
                    self.ctes()
                self.state = 305
                self.dmlStatementNoWith()
                pass
            elif la_ == 3:
                localctx = sqlParser.UseContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 306
                self.match(sqlParser.USE)
                self.state = 308
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 2, self._ctx)
                if la_ == 1:
                    self.state = 307
                    self.match(sqlParser.NAMESPACE)
                self.state = 310
                self.multipartIdentifier()
                pass
            elif la_ == 4:
                localctx = sqlParser.CreateNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 311
                self.match(sqlParser.CREATE)
                self.state = 312
                self.namespace()
                self.state = 316
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 3, self._ctx)
                if la_ == 1:
                    self.state = 313
                    self.match(sqlParser.IF)
                    self.state = 314
                    self.match(sqlParser.NOT)
                    self.state = 315
                    self.match(sqlParser.EXISTS)
                self.state = 318
                self.multipartIdentifier()
                self.state = 326
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 45 or _la == 138 or _la == 262:
                    self.state = 324
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [45]:
                        self.state = 319
                        self.commentSpec()
                        pass
                    elif token in [138]:
                        self.state = 320
                        self.locationSpec()
                        pass
                    elif token in [262]:
                        self.state = 321
                        self.match(sqlParser.WITH)
                        self.state = 322
                        _la = self._input.LA(1)
                        if not (_la == 65 or _la == 180):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 323
                        self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 328
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif la_ == 5:
                localctx = sqlParser.SetNamespacePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 329
                self.match(sqlParser.ALTER)
                self.state = 330
                self.namespace()
                self.state = 331
                self.multipartIdentifier()
                self.state = 332
                self.match(sqlParser.SET)
                self.state = 333
                _la = self._input.LA(1)
                if not (_la == 65 or _la == 180):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 334
                self.tablePropertyList()
                pass
            elif la_ == 6:
                localctx = sqlParser.SetNamespaceLocationContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 336
                self.match(sqlParser.ALTER)
                self.state = 337
                self.namespace()
                self.state = 338
                self.multipartIdentifier()
                self.state = 339
                self.match(sqlParser.SET)
                self.state = 340
                self.locationSpec()
                pass
            elif la_ == 7:
                localctx = sqlParser.DropNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 342
                self.match(sqlParser.DROP)
                self.state = 343
                self.namespace()
                self.state = 346
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 6, self._ctx)
                if la_ == 1:
                    self.state = 344
                    self.match(sqlParser.IF)
                    self.state = 345
                    self.match(sqlParser.EXISTS)
                self.state = 348
                self.multipartIdentifier()
                self.state = 350
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 32 or _la == 194:
                    self.state = 349
                    _la = self._input.LA(1)
                    if not (_la == 32 or _la == 194):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                pass
            elif la_ == 8:
                localctx = sqlParser.ShowNamespacesContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 352
                self.match(sqlParser.SHOW)
                self.state = 353
                _la = self._input.LA(1)
                if not (_la == 63 or _la == 150):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 356
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 100 or _la == 113:
                    self.state = 354
                    _la = self._input.LA(1)
                    if not (_la == 100 or _la == 113):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 355
                    self.multipartIdentifier()
                self.state = 362
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 132 or _la == 283:
                    self.state = 359
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 132:
                        self.state = 358
                        self.match(sqlParser.LIKE)
                    self.state = 361
                    localctx.pattern = self.match(sqlParser.STRING)
                pass
            elif la_ == 9:
                localctx = sqlParser.CreateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 364
                self.createTableHeader()
                self.state = 369
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 365
                    self.match(sqlParser.T__1)
                    self.state = 366
                    self.colTypeList()
                    self.state = 367
                    self.match(sqlParser.T__2)
                self.state = 371
                self.tableProvider()
                self.state = 372
                self.createTableClauses()
                self.state = 377
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2 or _la == 22 or _la == 100 or (_la == 143) or (_la - 187 & ~63 == 0 and 1 << _la - 187 & 1099512152065 != 0) or (_la == 256) or (_la == 262):
                    self.state = 374
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 22:
                        self.state = 373
                        self.match(sqlParser.AS)
                    self.state = 376
                    self.query()
                pass
            elif la_ == 10:
                localctx = sqlParser.CreateHiveTableContext(self, localctx)
                self.enterOuterAlt(localctx, 10)
                self.state = 379
                self.createTableHeader()
                self.state = 384
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 14, self._ctx)
                if la_ == 1:
                    self.state = 380
                    self.match(sqlParser.T__1)
                    self.state = 381
                    localctx.columns = self.colTypeList()
                    self.state = 382
                    self.match(sqlParser.T__2)
                self.state = 407
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 39 or _la == 45 or _la == 138 or (_la == 171) or (_la - 202 & ~63 == 0 and 1 << _la - 202 & 269500417 != 0):
                    self.state = 405
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [45]:
                        self.state = 386
                        self.commentSpec()
                        pass
                    elif token in [171]:
                        self.state = 396
                        self._errHandler.sync(self)
                        la_ = self._interp.adaptivePredict(self._input, 15, self._ctx)
                        if la_ == 1:
                            self.state = 387
                            self.match(sqlParser.PARTITIONED)
                            self.state = 388
                            self.match(sqlParser.BY)
                            self.state = 389
                            self.match(sqlParser.T__1)
                            self.state = 390
                            localctx.partitionColumns = self.colTypeList()
                            self.state = 391
                            self.match(sqlParser.T__2)
                            pass
                        elif la_ == 2:
                            self.state = 393
                            self.match(sqlParser.PARTITIONED)
                            self.state = 394
                            self.match(sqlParser.BY)
                            self.state = 395
                            localctx.partitionColumnNames = self.identifierList()
                            pass
                        pass
                    elif token in [39]:
                        self.state = 398
                        self.bucketSpec()
                        pass
                    elif token in [216]:
                        self.state = 399
                        self.skewSpec()
                        pass
                    elif token in [202]:
                        self.state = 400
                        self.rowFormat()
                        pass
                    elif token in [222]:
                        self.state = 401
                        self.createFileFormat()
                        pass
                    elif token in [138]:
                        self.state = 402
                        self.locationSpec()
                        pass
                    elif token in [230]:
                        self.state = 403
                        self.match(sqlParser.TBLPROPERTIES)
                        self.state = 404
                        localctx.tableProps = self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 409
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 414
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2 or _la == 22 or _la == 100 or (_la == 143) or (_la - 187 & ~63 == 0 and 1 << _la - 187 & 1099512152065 != 0) or (_la == 256) or (_la == 262):
                    self.state = 411
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 22:
                        self.state = 410
                        self.match(sqlParser.AS)
                    self.state = 413
                    self.query()
                pass
            elif la_ == 11:
                localctx = sqlParser.CreateTableLikeContext(self, localctx)
                self.enterOuterAlt(localctx, 11)
                self.state = 416
                self.match(sqlParser.CREATE)
                self.state = 417
                self.match(sqlParser.TABLE)
                self.state = 421
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 20, self._ctx)
                if la_ == 1:
                    self.state = 418
                    self.match(sqlParser.IF)
                    self.state = 419
                    self.match(sqlParser.NOT)
                    self.state = 420
                    self.match(sqlParser.EXISTS)
                self.state = 423
                localctx.target = self.tableIdentifier()
                self.state = 424
                self.match(sqlParser.LIKE)
                self.state = 425
                localctx.source = self.tableIdentifier()
                self.state = 434
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 138 or (_la - 202 & ~63 == 0 and 1 << _la - 202 & 9007199524225025 != 0):
                    self.state = 432
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [255]:
                        self.state = 426
                        self.tableProvider()
                        pass
                    elif token in [202]:
                        self.state = 427
                        self.rowFormat()
                        pass
                    elif token in [222]:
                        self.state = 428
                        self.createFileFormat()
                        pass
                    elif token in [138]:
                        self.state = 429
                        self.locationSpec()
                        pass
                    elif token in [230]:
                        self.state = 430
                        self.match(sqlParser.TBLPROPERTIES)
                        self.state = 431
                        localctx.tableProps = self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 436
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif la_ == 12:
                localctx = sqlParser.ReplaceTableContext(self, localctx)
                self.enterOuterAlt(localctx, 12)
                self.state = 437
                self.replaceTableHeader()
                self.state = 442
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 438
                    self.match(sqlParser.T__1)
                    self.state = 439
                    self.colTypeList()
                    self.state = 440
                    self.match(sqlParser.T__2)
                self.state = 444
                self.tableProvider()
                self.state = 445
                self.createTableClauses()
                self.state = 450
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2 or _la == 22 or _la == 100 or (_la == 143) or (_la - 187 & ~63 == 0 and 1 << _la - 187 & 1099512152065 != 0) or (_la == 256) or (_la == 262):
                    self.state = 447
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 22:
                        self.state = 446
                        self.match(sqlParser.AS)
                    self.state = 449
                    self.query()
                pass
            elif la_ == 13:
                localctx = sqlParser.AnalyzeContext(self, localctx)
                self.enterOuterAlt(localctx, 13)
                self.state = 452
                self.match(sqlParser.ANALYZE)
                self.state = 453
                self.match(sqlParser.TABLE)
                self.state = 454
                self.multipartIdentifier()
                self.state = 456
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 455
                    self.partitionSpec()
                self.state = 458
                self.match(sqlParser.COMPUTE)
                self.state = 459
                self.match(sqlParser.STATISTICS)
                self.state = 467
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 27, self._ctx)
                if la_ == 1:
                    self.state = 460
                    self.identifier()
                elif la_ == 2:
                    self.state = 461
                    self.match(sqlParser.FOR)
                    self.state = 462
                    self.match(sqlParser.COLUMNS)
                    self.state = 463
                    self.identifierSeq()
                elif la_ == 3:
                    self.state = 464
                    self.match(sqlParser.FOR)
                    self.state = 465
                    self.match(sqlParser.ALL)
                    self.state = 466
                    self.match(sqlParser.COLUMNS)
                pass
            elif la_ == 14:
                localctx = sqlParser.AddTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 14)
                self.state = 469
                self.match(sqlParser.ALTER)
                self.state = 470
                self.match(sqlParser.TABLE)
                self.state = 471
                self.multipartIdentifier()
                self.state = 472
                self.match(sqlParser.ADD)
                self.state = 473
                _la = self._input.LA(1)
                if not (_la == 43 or _la == 44):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 474
                localctx.columns = self.qualifiedColTypeWithPositionList()
                pass
            elif la_ == 15:
                localctx = sqlParser.AddTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 15)
                self.state = 476
                self.match(sqlParser.ALTER)
                self.state = 477
                self.match(sqlParser.TABLE)
                self.state = 478
                self.multipartIdentifier()
                self.state = 479
                self.match(sqlParser.ADD)
                self.state = 480
                _la = self._input.LA(1)
                if not (_la == 43 or _la == 44):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 481
                self.match(sqlParser.T__1)
                self.state = 482
                localctx.columns = self.qualifiedColTypeWithPositionList()
                self.state = 483
                self.match(sqlParser.T__2)
                pass
            elif la_ == 16:
                localctx = sqlParser.RenameTableColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 16)
                self.state = 485
                self.match(sqlParser.ALTER)
                self.state = 486
                self.match(sqlParser.TABLE)
                self.state = 487
                localctx.table = self.multipartIdentifier()
                self.state = 488
                self.match(sqlParser.RENAME)
                self.state = 489
                self.match(sqlParser.COLUMN)
                self.state = 490
                localctx.qpdfrom = self.multipartIdentifier()
                self.state = 491
                self.match(sqlParser.TO)
                self.state = 492
                localctx.to = self.errorCapturingIdentifier()
                pass
            elif la_ == 17:
                localctx = sqlParser.DropTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 17)
                self.state = 494
                self.match(sqlParser.ALTER)
                self.state = 495
                self.match(sqlParser.TABLE)
                self.state = 496
                self.multipartIdentifier()
                self.state = 497
                self.match(sqlParser.DROP)
                self.state = 498
                _la = self._input.LA(1)
                if not (_la == 43 or _la == 44):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 499
                self.match(sqlParser.T__1)
                self.state = 500
                localctx.columns = self.multipartIdentifierList()
                self.state = 501
                self.match(sqlParser.T__2)
                pass
            elif la_ == 18:
                localctx = sqlParser.DropTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 18)
                self.state = 503
                self.match(sqlParser.ALTER)
                self.state = 504
                self.match(sqlParser.TABLE)
                self.state = 505
                self.multipartIdentifier()
                self.state = 506
                self.match(sqlParser.DROP)
                self.state = 507
                _la = self._input.LA(1)
                if not (_la == 43 or _la == 44):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 508
                localctx.columns = self.multipartIdentifierList()
                pass
            elif la_ == 19:
                localctx = sqlParser.RenameTableContext(self, localctx)
                self.enterOuterAlt(localctx, 19)
                self.state = 510
                self.match(sqlParser.ALTER)
                self.state = 511
                _la = self._input.LA(1)
                if not (_la == 227 or _la == 257):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 512
                localctx.qpdfrom = self.multipartIdentifier()
                self.state = 513
                self.match(sqlParser.RENAME)
                self.state = 514
                self.match(sqlParser.TO)
                self.state = 515
                localctx.to = self.multipartIdentifier()
                pass
            elif la_ == 20:
                localctx = sqlParser.SetTablePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 20)
                self.state = 517
                self.match(sqlParser.ALTER)
                self.state = 518
                _la = self._input.LA(1)
                if not (_la == 227 or _la == 257):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 519
                self.multipartIdentifier()
                self.state = 520
                self.match(sqlParser.SET)
                self.state = 521
                self.match(sqlParser.TBLPROPERTIES)
                self.state = 522
                self.tablePropertyList()
                pass
            elif la_ == 21:
                localctx = sqlParser.UnsetTablePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 21)
                self.state = 524
                self.match(sqlParser.ALTER)
                self.state = 525
                _la = self._input.LA(1)
                if not (_la == 227 or _la == 257):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 526
                self.multipartIdentifier()
                self.state = 527
                self.match(sqlParser.UNSET)
                self.state = 528
                self.match(sqlParser.TBLPROPERTIES)
                self.state = 531
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 110:
                    self.state = 529
                    self.match(sqlParser.IF)
                    self.state = 530
                    self.match(sqlParser.EXISTS)
                self.state = 533
                self.tablePropertyList()
                pass
            elif la_ == 22:
                localctx = sqlParser.AlterTableAlterColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 22)
                self.state = 535
                self.match(sqlParser.ALTER)
                self.state = 536
                self.match(sqlParser.TABLE)
                self.state = 537
                localctx.table = self.multipartIdentifier()
                self.state = 538
                _la = self._input.LA(1)
                if not (_la == 15 or _la == 35):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 540
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 29, self._ctx)
                if la_ == 1:
                    self.state = 539
                    self.match(sqlParser.COLUMN)
                self.state = 542
                localctx.column = self.multipartIdentifier()
                self.state = 544
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 13 or _la == 45 or _la == 76 or (_la == 94) or (_la == 212) or (_la == 243):
                    self.state = 543
                    self.alterColumnAction()
                pass
            elif la_ == 23:
                localctx = sqlParser.HiveChangeColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 23)
                self.state = 546
                self.match(sqlParser.ALTER)
                self.state = 547
                self.match(sqlParser.TABLE)
                self.state = 548
                localctx.table = self.multipartIdentifier()
                self.state = 550
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 549
                    self.partitionSpec()
                self.state = 552
                self.match(sqlParser.CHANGE)
                self.state = 554
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 32, self._ctx)
                if la_ == 1:
                    self.state = 553
                    self.match(sqlParser.COLUMN)
                self.state = 556
                localctx.colName = self.multipartIdentifier()
                self.state = 557
                self.colType()
                self.state = 559
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 13 or _la == 94:
                    self.state = 558
                    self.colPosition()
                pass
            elif la_ == 24:
                localctx = sqlParser.HiveReplaceColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 24)
                self.state = 561
                self.match(sqlParser.ALTER)
                self.state = 562
                self.match(sqlParser.TABLE)
                self.state = 563
                localctx.table = self.multipartIdentifier()
                self.state = 565
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 564
                    self.partitionSpec()
                self.state = 567
                self.match(sqlParser.REPLACE)
                self.state = 568
                self.match(sqlParser.COLUMNS)
                self.state = 569
                self.match(sqlParser.T__1)
                self.state = 570
                localctx.columns = self.qualifiedColTypeWithPositionList()
                self.state = 571
                self.match(sqlParser.T__2)
                pass
            elif la_ == 25:
                localctx = sqlParser.SetTableSerDeContext(self, localctx)
                self.enterOuterAlt(localctx, 25)
                self.state = 573
                self.match(sqlParser.ALTER)
                self.state = 574
                self.match(sqlParser.TABLE)
                self.state = 575
                self.multipartIdentifier()
                self.state = 577
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 576
                    self.partitionSpec()
                self.state = 579
                self.match(sqlParser.SET)
                self.state = 580
                self.match(sqlParser.SERDE)
                self.state = 581
                self.match(sqlParser.STRING)
                self.state = 585
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 262:
                    self.state = 582
                    self.match(sqlParser.WITH)
                    self.state = 583
                    self.match(sqlParser.SERDEPROPERTIES)
                    self.state = 584
                    self.tablePropertyList()
                pass
            elif la_ == 26:
                localctx = sqlParser.SetTableSerDeContext(self, localctx)
                self.enterOuterAlt(localctx, 26)
                self.state = 587
                self.match(sqlParser.ALTER)
                self.state = 588
                self.match(sqlParser.TABLE)
                self.state = 589
                self.multipartIdentifier()
                self.state = 591
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 590
                    self.partitionSpec()
                self.state = 593
                self.match(sqlParser.SET)
                self.state = 594
                self.match(sqlParser.SERDEPROPERTIES)
                self.state = 595
                self.tablePropertyList()
                pass
            elif la_ == 27:
                localctx = sqlParser.AddTablePartitionContext(self, localctx)
                self.enterOuterAlt(localctx, 27)
                self.state = 597
                self.match(sqlParser.ALTER)
                self.state = 598
                _la = self._input.LA(1)
                if not (_la == 227 or _la == 257):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 599
                self.multipartIdentifier()
                self.state = 600
                self.match(sqlParser.ADD)
                self.state = 604
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 110:
                    self.state = 601
                    self.match(sqlParser.IF)
                    self.state = 602
                    self.match(sqlParser.NOT)
                    self.state = 603
                    self.match(sqlParser.EXISTS)
                self.state = 607
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 606
                    self.partitionSpecLocation()
                    self.state = 609
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 170:
                        break
                pass
            elif la_ == 28:
                localctx = sqlParser.RenameTablePartitionContext(self, localctx)
                self.enterOuterAlt(localctx, 28)
                self.state = 611
                self.match(sqlParser.ALTER)
                self.state = 612
                self.match(sqlParser.TABLE)
                self.state = 613
                self.multipartIdentifier()
                self.state = 614
                localctx.qpdfrom = self.partitionSpec()
                self.state = 615
                self.match(sqlParser.RENAME)
                self.state = 616
                self.match(sqlParser.TO)
                self.state = 617
                localctx.to = self.partitionSpec()
                pass
            elif la_ == 29:
                localctx = sqlParser.DropTablePartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 29)
                self.state = 619
                self.match(sqlParser.ALTER)
                self.state = 620
                _la = self._input.LA(1)
                if not (_la == 227 or _la == 257):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 621
                self.multipartIdentifier()
                self.state = 622
                self.match(sqlParser.DROP)
                self.state = 625
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 110:
                    self.state = 623
                    self.match(sqlParser.IF)
                    self.state = 624
                    self.match(sqlParser.EXISTS)
                self.state = 627
                self.partitionSpec()
                self.state = 632
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 628
                    self.match(sqlParser.T__3)
                    self.state = 629
                    self.partitionSpec()
                    self.state = 634
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 636
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 181:
                    self.state = 635
                    self.match(sqlParser.PURGE)
                pass
            elif la_ == 30:
                localctx = sqlParser.SetTableLocationContext(self, localctx)
                self.enterOuterAlt(localctx, 30)
                self.state = 638
                self.match(sqlParser.ALTER)
                self.state = 639
                self.match(sqlParser.TABLE)
                self.state = 640
                self.multipartIdentifier()
                self.state = 642
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 641
                    self.partitionSpec()
                self.state = 644
                self.match(sqlParser.SET)
                self.state = 645
                self.locationSpec()
                pass
            elif la_ == 31:
                localctx = sqlParser.RecoverPartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 31)
                self.state = 647
                self.match(sqlParser.ALTER)
                self.state = 648
                self.match(sqlParser.TABLE)
                self.state = 649
                self.multipartIdentifier()
                self.state = 650
                self.match(sqlParser.RECOVER)
                self.state = 651
                self.match(sqlParser.PARTITIONS)
                pass
            elif la_ == 32:
                localctx = sqlParser.DropTableContext(self, localctx)
                self.enterOuterAlt(localctx, 32)
                self.state = 653
                self.match(sqlParser.DROP)
                self.state = 654
                self.match(sqlParser.TABLE)
                self.state = 657
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 44, self._ctx)
                if la_ == 1:
                    self.state = 655
                    self.match(sqlParser.IF)
                    self.state = 656
                    self.match(sqlParser.EXISTS)
                self.state = 659
                self.multipartIdentifier()
                self.state = 661
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 181:
                    self.state = 660
                    self.match(sqlParser.PURGE)
                pass
            elif la_ == 33:
                localctx = sqlParser.DropViewContext(self, localctx)
                self.enterOuterAlt(localctx, 33)
                self.state = 663
                self.match(sqlParser.DROP)
                self.state = 664
                self.match(sqlParser.VIEW)
                self.state = 667
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 46, self._ctx)
                if la_ == 1:
                    self.state = 665
                    self.match(sqlParser.IF)
                    self.state = 666
                    self.match(sqlParser.EXISTS)
                self.state = 669
                self.multipartIdentifier()
                pass
            elif la_ == 34:
                localctx = sqlParser.CreateViewContext(self, localctx)
                self.enterOuterAlt(localctx, 34)
                self.state = 670
                self.match(sqlParser.CREATE)
                self.state = 673
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 161:
                    self.state = 671
                    self.match(sqlParser.OR)
                    self.state = 672
                    self.match(sqlParser.REPLACE)
                self.state = 679
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 104 or _la == 231:
                    self.state = 676
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 104:
                        self.state = 675
                        self.match(sqlParser.GLOBAL)
                    self.state = 678
                    self.match(sqlParser.TEMPORARY)
                self.state = 681
                self.match(sqlParser.VIEW)
                self.state = 685
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 50, self._ctx)
                if la_ == 1:
                    self.state = 682
                    self.match(sqlParser.IF)
                    self.state = 683
                    self.match(sqlParser.NOT)
                    self.state = 684
                    self.match(sqlParser.EXISTS)
                self.state = 687
                self.multipartIdentifier()
                self.state = 689
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 688
                    self.identifierCommentList()
                self.state = 699
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 45 or _la == 171 or _la == 230:
                    self.state = 697
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [45]:
                        self.state = 691
                        self.commentSpec()
                        pass
                    elif token in [171]:
                        self.state = 692
                        self.match(sqlParser.PARTITIONED)
                        self.state = 693
                        self.match(sqlParser.ON)
                        self.state = 694
                        self.identifierList()
                        pass
                    elif token in [230]:
                        self.state = 695
                        self.match(sqlParser.TBLPROPERTIES)
                        self.state = 696
                        self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 701
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 702
                self.match(sqlParser.AS)
                self.state = 703
                self.query()
                pass
            elif la_ == 35:
                localctx = sqlParser.CreateTempViewUsingContext(self, localctx)
                self.enterOuterAlt(localctx, 35)
                self.state = 705
                self.match(sqlParser.CREATE)
                self.state = 708
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 161:
                    self.state = 706
                    self.match(sqlParser.OR)
                    self.state = 707
                    self.match(sqlParser.REPLACE)
                self.state = 711
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 104:
                    self.state = 710
                    self.match(sqlParser.GLOBAL)
                self.state = 713
                self.match(sqlParser.TEMPORARY)
                self.state = 714
                self.match(sqlParser.VIEW)
                self.state = 715
                self.tableIdentifier()
                self.state = 720
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 716
                    self.match(sqlParser.T__1)
                    self.state = 717
                    self.colTypeList()
                    self.state = 718
                    self.match(sqlParser.T__2)
                self.state = 722
                self.tableProvider()
                self.state = 725
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 160:
                    self.state = 723
                    self.match(sqlParser.OPTIONS)
                    self.state = 724
                    self.tablePropertyList()
                pass
            elif la_ == 36:
                localctx = sqlParser.AlterViewQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 36)
                self.state = 727
                self.match(sqlParser.ALTER)
                self.state = 728
                self.match(sqlParser.VIEW)
                self.state = 729
                self.multipartIdentifier()
                self.state = 731
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 22:
                    self.state = 730
                    self.match(sqlParser.AS)
                self.state = 733
                self.query()
                pass
            elif la_ == 37:
                localctx = sqlParser.CreateFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 37)
                self.state = 735
                self.match(sqlParser.CREATE)
                self.state = 738
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 161:
                    self.state = 736
                    self.match(sqlParser.OR)
                    self.state = 737
                    self.match(sqlParser.REPLACE)
                self.state = 741
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 231:
                    self.state = 740
                    self.match(sqlParser.TEMPORARY)
                self.state = 743
                self.match(sqlParser.FUNCTION)
                self.state = 747
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 61, self._ctx)
                if la_ == 1:
                    self.state = 744
                    self.match(sqlParser.IF)
                    self.state = 745
                    self.match(sqlParser.NOT)
                    self.state = 746
                    self.match(sqlParser.EXISTS)
                self.state = 749
                self.multipartIdentifier()
                self.state = 750
                self.match(sqlParser.AS)
                self.state = 751
                localctx.className = self.match(sqlParser.STRING)
                self.state = 761
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 255:
                    self.state = 752
                    self.match(sqlParser.USING)
                    self.state = 753
                    self.resource()
                    self.state = 758
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 754
                        self.match(sqlParser.T__3)
                        self.state = 755
                        self.resource()
                        self.state = 760
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                pass
            elif la_ == 38:
                localctx = sqlParser.DropFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 38)
                self.state = 763
                self.match(sqlParser.DROP)
                self.state = 765
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 231:
                    self.state = 764
                    self.match(sqlParser.TEMPORARY)
                self.state = 767
                self.match(sqlParser.FUNCTION)
                self.state = 770
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 65, self._ctx)
                if la_ == 1:
                    self.state = 768
                    self.match(sqlParser.IF)
                    self.state = 769
                    self.match(sqlParser.EXISTS)
                self.state = 772
                self.multipartIdentifier()
                pass
            elif la_ == 39:
                localctx = sqlParser.ExplainContext(self, localctx)
                self.enterOuterAlt(localctx, 39)
                self.state = 773
                self.match(sqlParser.EXPLAIN)
                self.state = 775
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 40 or _la == 52 or (_la - 86 & ~63 == 0 and 1 << _la - 86 & 36028797018972161 != 0):
                    self.state = 774
                    _la = self._input.LA(1)
                    if not (_la == 40 or _la == 52 or (_la - 86 & ~63 == 0 and 1 << _la - 86 & 36028797018972161 != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 777
                self.statement()
                pass
            elif la_ == 40:
                localctx = sqlParser.ShowTablesContext(self, localctx)
                self.enterOuterAlt(localctx, 40)
                self.state = 778
                self.match(sqlParser.SHOW)
                self.state = 779
                self.match(sqlParser.TABLES)
                self.state = 782
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 100 or _la == 113:
                    self.state = 780
                    _la = self._input.LA(1)
                    if not (_la == 100 or _la == 113):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 781
                    self.multipartIdentifier()
                self.state = 788
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 132 or _la == 283:
                    self.state = 785
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 132:
                        self.state = 784
                        self.match(sqlParser.LIKE)
                    self.state = 787
                    localctx.pattern = self.match(sqlParser.STRING)
                pass
            elif la_ == 41:
                localctx = sqlParser.ShowTableContext(self, localctx)
                self.enterOuterAlt(localctx, 41)
                self.state = 790
                self.match(sqlParser.SHOW)
                self.state = 791
                self.match(sqlParser.TABLE)
                self.state = 792
                self.match(sqlParser.EXTENDED)
                self.state = 795
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 100 or _la == 113:
                    self.state = 793
                    _la = self._input.LA(1)
                    if not (_la == 100 or _la == 113):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 794
                    localctx.ns = self.multipartIdentifier()
                self.state = 797
                self.match(sqlParser.LIKE)
                self.state = 798
                localctx.pattern = self.match(sqlParser.STRING)
                self.state = 800
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 799
                    self.partitionSpec()
                pass
            elif la_ == 42:
                localctx = sqlParser.ShowTblPropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 42)
                self.state = 802
                self.match(sqlParser.SHOW)
                self.state = 803
                self.match(sqlParser.TBLPROPERTIES)
                self.state = 804
                localctx.table = self.multipartIdentifier()
                self.state = 809
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 805
                    self.match(sqlParser.T__1)
                    self.state = 806
                    localctx.key = self.tablePropertyKey()
                    self.state = 807
                    self.match(sqlParser.T__2)
                pass
            elif la_ == 43:
                localctx = sqlParser.ShowColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 43)
                self.state = 811
                self.match(sqlParser.SHOW)
                self.state = 812
                self.match(sqlParser.COLUMNS)
                self.state = 813
                _la = self._input.LA(1)
                if not (_la == 100 or _la == 113):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 814
                localctx.table = self.multipartIdentifier()
                self.state = 817
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 100 or _la == 113:
                    self.state = 815
                    _la = self._input.LA(1)
                    if not (_la == 100 or _la == 113):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 816
                    localctx.ns = self.multipartIdentifier()
                pass
            elif la_ == 44:
                localctx = sqlParser.ShowViewsContext(self, localctx)
                self.enterOuterAlt(localctx, 44)
                self.state = 819
                self.match(sqlParser.SHOW)
                self.state = 820
                self.match(sqlParser.VIEWS)
                self.state = 823
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 100 or _la == 113:
                    self.state = 821
                    _la = self._input.LA(1)
                    if not (_la == 100 or _la == 113):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 822
                    self.multipartIdentifier()
                self.state = 829
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 132 or _la == 283:
                    self.state = 826
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 132:
                        self.state = 825
                        self.match(sqlParser.LIKE)
                    self.state = 828
                    localctx.pattern = self.match(sqlParser.STRING)
                pass
            elif la_ == 45:
                localctx = sqlParser.ShowPartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 45)
                self.state = 831
                self.match(sqlParser.SHOW)
                self.state = 832
                self.match(sqlParser.PARTITIONS)
                self.state = 833
                self.multipartIdentifier()
                self.state = 835
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 834
                    self.partitionSpec()
                pass
            elif la_ == 46:
                localctx = sqlParser.ShowFunctionsContext(self, localctx)
                self.enterOuterAlt(localctx, 46)
                self.state = 837
                self.match(sqlParser.SHOW)
                self.state = 839
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 78, self._ctx)
                if la_ == 1:
                    self.state = 838
                    self.identifier()
                self.state = 841
                self.match(sqlParser.FUNCTIONS)
                self.state = 849
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 81, self._ctx)
                if la_ == 1:
                    self.state = 843
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 79, self._ctx)
                    if la_ == 1:
                        self.state = 842
                        self.match(sqlParser.LIKE)
                    self.state = 847
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 80, self._ctx)
                    if la_ == 1:
                        self.state = 845
                        self.multipartIdentifier()
                        pass
                    elif la_ == 2:
                        self.state = 846
                        localctx.pattern = self.match(sqlParser.STRING)
                        pass
                pass
            elif la_ == 47:
                localctx = sqlParser.ShowCreateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 47)
                self.state = 851
                self.match(sqlParser.SHOW)
                self.state = 852
                self.match(sqlParser.CREATE)
                self.state = 853
                self.match(sqlParser.TABLE)
                self.state = 854
                self.multipartIdentifier()
                self.state = 857
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 22:
                    self.state = 855
                    self.match(sqlParser.AS)
                    self.state = 856
                    self.match(sqlParser.SERDE)
                pass
            elif la_ == 48:
                localctx = sqlParser.ShowCurrentNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 48)
                self.state = 859
                self.match(sqlParser.SHOW)
                self.state = 860
                self.match(sqlParser.CURRENT)
                self.state = 861
                self.match(sqlParser.NAMESPACE)
                pass
            elif la_ == 49:
                localctx = sqlParser.DescribeFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 49)
                self.state = 862
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 70):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 863
                self.match(sqlParser.FUNCTION)
                self.state = 865
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 83, self._ctx)
                if la_ == 1:
                    self.state = 864
                    self.match(sqlParser.EXTENDED)
                self.state = 867
                self.describeFuncName()
                pass
            elif la_ == 50:
                localctx = sqlParser.DescribeNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 50)
                self.state = 868
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 70):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 869
                self.namespace()
                self.state = 871
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 84, self._ctx)
                if la_ == 1:
                    self.state = 870
                    self.match(sqlParser.EXTENDED)
                self.state = 873
                self.multipartIdentifier()
                pass
            elif la_ == 51:
                localctx = sqlParser.DescribeRelationContext(self, localctx)
                self.enterOuterAlt(localctx, 51)
                self.state = 875
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 70):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 877
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 85, self._ctx)
                if la_ == 1:
                    self.state = 876
                    self.match(sqlParser.TABLE)
                self.state = 880
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 86, self._ctx)
                if la_ == 1:
                    self.state = 879
                    localctx.option = self._input.LT(1)
                    _la = self._input.LA(1)
                    if not (_la == 86 or _la == 99):
                        localctx.option = self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 882
                self.multipartIdentifier()
                self.state = 884
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 87, self._ctx)
                if la_ == 1:
                    self.state = 883
                    self.partitionSpec()
                self.state = 887
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 88, self._ctx)
                if la_ == 1:
                    self.state = 886
                    self.describeColName()
                pass
            elif la_ == 52:
                localctx = sqlParser.DescribeQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 52)
                self.state = 889
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 70):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 891
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 182:
                    self.state = 890
                    self.match(sqlParser.QUERY)
                self.state = 893
                self.query()
                pass
            elif la_ == 53:
                localctx = sqlParser.CommentNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 53)
                self.state = 894
                self.match(sqlParser.COMMENT)
                self.state = 895
                self.match(sqlParser.ON)
                self.state = 896
                self.namespace()
                self.state = 897
                self.multipartIdentifier()
                self.state = 898
                self.match(sqlParser.IS)
                self.state = 899
                localctx.comment = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 154 or _la == 283):
                    localctx.comment = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 54:
                localctx = sqlParser.CommentTableContext(self, localctx)
                self.enterOuterAlt(localctx, 54)
                self.state = 901
                self.match(sqlParser.COMMENT)
                self.state = 902
                self.match(sqlParser.ON)
                self.state = 903
                self.match(sqlParser.TABLE)
                self.state = 904
                self.multipartIdentifier()
                self.state = 905
                self.match(sqlParser.IS)
                self.state = 906
                localctx.comment = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 154 or _la == 283):
                    localctx.comment = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 55:
                localctx = sqlParser.RefreshTableContext(self, localctx)
                self.enterOuterAlt(localctx, 55)
                self.state = 908
                self.match(sqlParser.REFRESH)
                self.state = 909
                self.match(sqlParser.TABLE)
                self.state = 910
                self.multipartIdentifier()
                pass
            elif la_ == 56:
                localctx = sqlParser.RefreshResourceContext(self, localctx)
                self.enterOuterAlt(localctx, 56)
                self.state = 911
                self.match(sqlParser.REFRESH)
                self.state = 919
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 91, self._ctx)
                if la_ == 1:
                    self.state = 912
                    self.match(sqlParser.STRING)
                    pass
                elif la_ == 2:
                    self.state = 916
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 90, self._ctx)
                    while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                        if _alt == 1 + 1:
                            self.state = 913
                            self.matchWildcard()
                        self.state = 918
                        self._errHandler.sync(self)
                        _alt = self._interp.adaptivePredict(self._input, 90, self._ctx)
                    pass
                pass
            elif la_ == 57:
                localctx = sqlParser.CacheTableContext(self, localctx)
                self.enterOuterAlt(localctx, 57)
                self.state = 921
                self.match(sqlParser.CACHE)
                self.state = 923
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 129:
                    self.state = 922
                    self.match(sqlParser.LAZY)
                self.state = 925
                self.match(sqlParser.TABLE)
                self.state = 926
                self.multipartIdentifier()
                self.state = 929
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 160:
                    self.state = 927
                    self.match(sqlParser.OPTIONS)
                    self.state = 928
                    localctx.options = self.tablePropertyList()
                self.state = 935
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2 or _la == 22 or _la == 100 or (_la == 143) or (_la - 187 & ~63 == 0 and 1 << _la - 187 & 1099512152065 != 0) or (_la == 256) or (_la == 262):
                    self.state = 932
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 22:
                        self.state = 931
                        self.match(sqlParser.AS)
                    self.state = 934
                    self.query()
                pass
            elif la_ == 58:
                localctx = sqlParser.UncacheTableContext(self, localctx)
                self.enterOuterAlt(localctx, 58)
                self.state = 937
                self.match(sqlParser.UNCACHE)
                self.state = 938
                self.match(sqlParser.TABLE)
                self.state = 941
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 96, self._ctx)
                if la_ == 1:
                    self.state = 939
                    self.match(sqlParser.IF)
                    self.state = 940
                    self.match(sqlParser.EXISTS)
                self.state = 943
                self.multipartIdentifier()
                pass
            elif la_ == 59:
                localctx = sqlParser.ClearCacheContext(self, localctx)
                self.enterOuterAlt(localctx, 59)
                self.state = 944
                self.match(sqlParser.CLEAR)
                self.state = 945
                self.match(sqlParser.CACHE)
                pass
            elif la_ == 60:
                localctx = sqlParser.LoadDataContext(self, localctx)
                self.enterOuterAlt(localctx, 60)
                self.state = 946
                self.match(sqlParser.LOAD)
                self.state = 947
                self.match(sqlParser.DATA)
                self.state = 949
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 137:
                    self.state = 948
                    self.match(sqlParser.LOCAL)
                self.state = 951
                self.match(sqlParser.INPATH)
                self.state = 952
                localctx.path = self.match(sqlParser.STRING)
                self.state = 954
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 169:
                    self.state = 953
                    self.match(sqlParser.OVERWRITE)
                self.state = 956
                self.match(sqlParser.INTO)
                self.state = 957
                self.match(sqlParser.TABLE)
                self.state = 958
                self.multipartIdentifier()
                self.state = 960
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 959
                    self.partitionSpec()
                pass
            elif la_ == 61:
                localctx = sqlParser.TruncateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 61)
                self.state = 962
                self.match(sqlParser.TRUNCATE)
                self.state = 963
                self.match(sqlParser.TABLE)
                self.state = 964
                self.multipartIdentifier()
                self.state = 966
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 965
                    self.partitionSpec()
                pass
            elif la_ == 62:
                localctx = sqlParser.RepairTableContext(self, localctx)
                self.enterOuterAlt(localctx, 62)
                self.state = 968
                self.match(sqlParser.MSCK)
                self.state = 969
                self.match(sqlParser.REPAIR)
                self.state = 970
                self.match(sqlParser.TABLE)
                self.state = 971
                self.multipartIdentifier()
                pass
            elif la_ == 63:
                localctx = sqlParser.ManageResourceContext(self, localctx)
                self.enterOuterAlt(localctx, 63)
                self.state = 972
                localctx.op = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 12 or _la == 135):
                    localctx.op = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 973
                self.identifier()
                self.state = 981
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 102, self._ctx)
                if la_ == 1:
                    self.state = 974
                    self.match(sqlParser.STRING)
                    pass
                elif la_ == 2:
                    self.state = 978
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 101, self._ctx)
                    while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                        if _alt == 1 + 1:
                            self.state = 975
                            self.matchWildcard()
                        self.state = 980
                        self._errHandler.sync(self)
                        _alt = self._interp.adaptivePredict(self._input, 101, self._ctx)
                    pass
                pass
            elif la_ == 64:
                localctx = sqlParser.FailNativeCommandContext(self, localctx)
                self.enterOuterAlt(localctx, 64)
                self.state = 983
                self.match(sqlParser.SET)
                self.state = 984
                self.match(sqlParser.ROLE)
                self.state = 988
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 103, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 985
                        self.matchWildcard()
                    self.state = 990
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 103, self._ctx)
                pass
            elif la_ == 65:
                localctx = sqlParser.SetConfigurationContext(self, localctx)
                self.enterOuterAlt(localctx, 65)
                self.state = 991
                self.match(sqlParser.SET)
                self.state = 995
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 104, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 992
                        self.matchWildcard()
                    self.state = 997
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 104, self._ctx)
                pass
            elif la_ == 66:
                localctx = sqlParser.ResetConfigurationContext(self, localctx)
                self.enterOuterAlt(localctx, 66)
                self.state = 998
                self.match(sqlParser.RESET)
                pass
            elif la_ == 67:
                localctx = sqlParser.FailNativeCommandContext(self, localctx)
                self.enterOuterAlt(localctx, 67)
                self.state = 999
                self.unsupportedHiveNativeCommands()
                self.state = 1003
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 105, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 1000
                        self.matchWildcard()
                    self.state = 1005
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 105, self._ctx)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class UnsupportedHiveNativeCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.kw1 = None
            self.kw2 = None
            self.kw3 = None
            self.kw4 = None
            self.kw5 = None
            self.kw6 = None

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def ROLE(self):
            return self.getToken(sqlParser.ROLE, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def GRANT(self):
            return self.getToken(sqlParser.GRANT, 0)

        def REVOKE(self):
            return self.getToken(sqlParser.REVOKE, 0)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def PRINCIPALS(self):
            return self.getToken(sqlParser.PRINCIPALS, 0)

        def ROLES(self):
            return self.getToken(sqlParser.ROLES, 0)

        def CURRENT(self):
            return self.getToken(sqlParser.CURRENT, 0)

        def EXPORT(self):
            return self.getToken(sqlParser.EXPORT, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def IMPORT(self):
            return self.getToken(sqlParser.IMPORT, 0)

        def COMPACTIONS(self):
            return self.getToken(sqlParser.COMPACTIONS, 0)

        def TRANSACTIONS(self):
            return self.getToken(sqlParser.TRANSACTIONS, 0)

        def INDEXES(self):
            return self.getToken(sqlParser.INDEXES, 0)

        def LOCKS(self):
            return self.getToken(sqlParser.LOCKS, 0)

        def INDEX(self):
            return self.getToken(sqlParser.INDEX, 0)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def LOCK(self):
            return self.getToken(sqlParser.LOCK, 0)

        def DATABASE(self):
            return self.getToken(sqlParser.DATABASE, 0)

        def UNLOCK(self):
            return self.getToken(sqlParser.UNLOCK, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def MACRO(self):
            return self.getToken(sqlParser.MACRO, 0)

        def tableIdentifier(self):
            return self.getTypedRuleContext(sqlParser.TableIdentifierContext, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def CLUSTERED(self):
            return self.getToken(sqlParser.CLUSTERED, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def SORTED(self):
            return self.getToken(sqlParser.SORTED, 0)

        def SKEWED(self):
            return self.getToken(sqlParser.SKEWED, 0)

        def STORED(self):
            return self.getToken(sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def DIRECTORIES(self):
            return self.getToken(sqlParser.DIRECTORIES, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def LOCATION(self):
            return self.getToken(sqlParser.LOCATION, 0)

        def EXCHANGE(self):
            return self.getToken(sqlParser.EXCHANGE, 0)

        def PARTITION(self):
            return self.getToken(sqlParser.PARTITION, 0)

        def ARCHIVE(self):
            return self.getToken(sqlParser.ARCHIVE, 0)

        def UNARCHIVE(self):
            return self.getToken(sqlParser.UNARCHIVE, 0)

        def TOUCH(self):
            return self.getToken(sqlParser.TOUCH, 0)

        def COMPACT(self):
            return self.getToken(sqlParser.COMPACT, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def CONCATENATE(self):
            return self.getToken(sqlParser.CONCATENATE, 0)

        def FILEFORMAT(self):
            return self.getToken(sqlParser.FILEFORMAT, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def START(self):
            return self.getToken(sqlParser.START, 0)

        def TRANSACTION(self):
            return self.getToken(sqlParser.TRANSACTION, 0)

        def COMMIT(self):
            return self.getToken(sqlParser.COMMIT, 0)

        def ROLLBACK(self):
            return self.getToken(sqlParser.ROLLBACK, 0)

        def DFS(self):
            return self.getToken(sqlParser.DFS, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_unsupportedHiveNativeCommands

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnsupportedHiveNativeCommands'):
                return visitor.visitUnsupportedHiveNativeCommands(self)
            else:
                return visitor.visitChildren(self)

    def unsupportedHiveNativeCommands(self):
        localctx = sqlParser.UnsupportedHiveNativeCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_unsupportedHiveNativeCommands)
        self._la = 0
        try:
            self.state = 1176
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 114, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1008
                localctx.kw1 = self.match(sqlParser.CREATE)
                self.state = 1009
                localctx.kw2 = self.match(sqlParser.ROLE)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1010
                localctx.kw1 = self.match(sqlParser.DROP)
                self.state = 1011
                localctx.kw2 = self.match(sqlParser.ROLE)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 1012
                localctx.kw1 = self.match(sqlParser.GRANT)
                self.state = 1014
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 107, self._ctx)
                if la_ == 1:
                    self.state = 1013
                    localctx.kw2 = self.match(sqlParser.ROLE)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 1016
                localctx.kw1 = self.match(sqlParser.REVOKE)
                self.state = 1018
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 108, self._ctx)
                if la_ == 1:
                    self.state = 1017
                    localctx.kw2 = self.match(sqlParser.ROLE)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 1020
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1021
                localctx.kw2 = self.match(sqlParser.GRANT)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 1022
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1023
                localctx.kw2 = self.match(sqlParser.ROLE)
                self.state = 1025
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 109, self._ctx)
                if la_ == 1:
                    self.state = 1024
                    localctx.kw3 = self.match(sqlParser.GRANT)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 1027
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1028
                localctx.kw2 = self.match(sqlParser.PRINCIPALS)
                pass
            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 1029
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1030
                localctx.kw2 = self.match(sqlParser.ROLES)
                pass
            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 1031
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1032
                localctx.kw2 = self.match(sqlParser.CURRENT)
                self.state = 1033
                localctx.kw3 = self.match(sqlParser.ROLES)
                pass
            elif la_ == 10:
                self.enterOuterAlt(localctx, 10)
                self.state = 1034
                localctx.kw1 = self.match(sqlParser.EXPORT)
                self.state = 1035
                localctx.kw2 = self.match(sqlParser.TABLE)
                pass
            elif la_ == 11:
                self.enterOuterAlt(localctx, 11)
                self.state = 1036
                localctx.kw1 = self.match(sqlParser.IMPORT)
                self.state = 1037
                localctx.kw2 = self.match(sqlParser.TABLE)
                pass
            elif la_ == 12:
                self.enterOuterAlt(localctx, 12)
                self.state = 1038
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1039
                localctx.kw2 = self.match(sqlParser.COMPACTIONS)
                pass
            elif la_ == 13:
                self.enterOuterAlt(localctx, 13)
                self.state = 1040
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1041
                localctx.kw2 = self.match(sqlParser.CREATE)
                self.state = 1042
                localctx.kw3 = self.match(sqlParser.TABLE)
                pass
            elif la_ == 14:
                self.enterOuterAlt(localctx, 14)
                self.state = 1043
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1044
                localctx.kw2 = self.match(sqlParser.TRANSACTIONS)
                pass
            elif la_ == 15:
                self.enterOuterAlt(localctx, 15)
                self.state = 1045
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1046
                localctx.kw2 = self.match(sqlParser.INDEXES)
                pass
            elif la_ == 16:
                self.enterOuterAlt(localctx, 16)
                self.state = 1047
                localctx.kw1 = self.match(sqlParser.SHOW)
                self.state = 1048
                localctx.kw2 = self.match(sqlParser.LOCKS)
                pass
            elif la_ == 17:
                self.enterOuterAlt(localctx, 17)
                self.state = 1049
                localctx.kw1 = self.match(sqlParser.CREATE)
                self.state = 1050
                localctx.kw2 = self.match(sqlParser.INDEX)
                pass
            elif la_ == 18:
                self.enterOuterAlt(localctx, 18)
                self.state = 1051
                localctx.kw1 = self.match(sqlParser.DROP)
                self.state = 1052
                localctx.kw2 = self.match(sqlParser.INDEX)
                pass
            elif la_ == 19:
                self.enterOuterAlt(localctx, 19)
                self.state = 1053
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1054
                localctx.kw2 = self.match(sqlParser.INDEX)
                pass
            elif la_ == 20:
                self.enterOuterAlt(localctx, 20)
                self.state = 1055
                localctx.kw1 = self.match(sqlParser.LOCK)
                self.state = 1056
                localctx.kw2 = self.match(sqlParser.TABLE)
                pass
            elif la_ == 21:
                self.enterOuterAlt(localctx, 21)
                self.state = 1057
                localctx.kw1 = self.match(sqlParser.LOCK)
                self.state = 1058
                localctx.kw2 = self.match(sqlParser.DATABASE)
                pass
            elif la_ == 22:
                self.enterOuterAlt(localctx, 22)
                self.state = 1059
                localctx.kw1 = self.match(sqlParser.UNLOCK)
                self.state = 1060
                localctx.kw2 = self.match(sqlParser.TABLE)
                pass
            elif la_ == 23:
                self.enterOuterAlt(localctx, 23)
                self.state = 1061
                localctx.kw1 = self.match(sqlParser.UNLOCK)
                self.state = 1062
                localctx.kw2 = self.match(sqlParser.DATABASE)
                pass
            elif la_ == 24:
                self.enterOuterAlt(localctx, 24)
                self.state = 1063
                localctx.kw1 = self.match(sqlParser.CREATE)
                self.state = 1064
                localctx.kw2 = self.match(sqlParser.TEMPORARY)
                self.state = 1065
                localctx.kw3 = self.match(sqlParser.MACRO)
                pass
            elif la_ == 25:
                self.enterOuterAlt(localctx, 25)
                self.state = 1066
                localctx.kw1 = self.match(sqlParser.DROP)
                self.state = 1067
                localctx.kw2 = self.match(sqlParser.TEMPORARY)
                self.state = 1068
                localctx.kw3 = self.match(sqlParser.MACRO)
                pass
            elif la_ == 26:
                self.enterOuterAlt(localctx, 26)
                self.state = 1069
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1070
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1071
                self.tableIdentifier()
                self.state = 1072
                localctx.kw3 = self.match(sqlParser.NOT)
                self.state = 1073
                localctx.kw4 = self.match(sqlParser.CLUSTERED)
                pass
            elif la_ == 27:
                self.enterOuterAlt(localctx, 27)
                self.state = 1075
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1076
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1077
                self.tableIdentifier()
                self.state = 1078
                localctx.kw3 = self.match(sqlParser.CLUSTERED)
                self.state = 1079
                localctx.kw4 = self.match(sqlParser.BY)
                pass
            elif la_ == 28:
                self.enterOuterAlt(localctx, 28)
                self.state = 1081
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1082
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1083
                self.tableIdentifier()
                self.state = 1084
                localctx.kw3 = self.match(sqlParser.NOT)
                self.state = 1085
                localctx.kw4 = self.match(sqlParser.SORTED)
                pass
            elif la_ == 29:
                self.enterOuterAlt(localctx, 29)
                self.state = 1087
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1088
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1089
                self.tableIdentifier()
                self.state = 1090
                localctx.kw3 = self.match(sqlParser.SKEWED)
                self.state = 1091
                localctx.kw4 = self.match(sqlParser.BY)
                pass
            elif la_ == 30:
                self.enterOuterAlt(localctx, 30)
                self.state = 1093
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1094
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1095
                self.tableIdentifier()
                self.state = 1096
                localctx.kw3 = self.match(sqlParser.NOT)
                self.state = 1097
                localctx.kw4 = self.match(sqlParser.SKEWED)
                pass
            elif la_ == 31:
                self.enterOuterAlt(localctx, 31)
                self.state = 1099
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1100
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1101
                self.tableIdentifier()
                self.state = 1102
                localctx.kw3 = self.match(sqlParser.NOT)
                self.state = 1103
                localctx.kw4 = self.match(sqlParser.STORED)
                self.state = 1104
                localctx.kw5 = self.match(sqlParser.AS)
                self.state = 1105
                localctx.kw6 = self.match(sqlParser.DIRECTORIES)
                pass
            elif la_ == 32:
                self.enterOuterAlt(localctx, 32)
                self.state = 1107
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1108
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1109
                self.tableIdentifier()
                self.state = 1110
                localctx.kw3 = self.match(sqlParser.SET)
                self.state = 1111
                localctx.kw4 = self.match(sqlParser.SKEWED)
                self.state = 1112
                localctx.kw5 = self.match(sqlParser.LOCATION)
                pass
            elif la_ == 33:
                self.enterOuterAlt(localctx, 33)
                self.state = 1114
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1115
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1116
                self.tableIdentifier()
                self.state = 1117
                localctx.kw3 = self.match(sqlParser.EXCHANGE)
                self.state = 1118
                localctx.kw4 = self.match(sqlParser.PARTITION)
                pass
            elif la_ == 34:
                self.enterOuterAlt(localctx, 34)
                self.state = 1120
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1121
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1122
                self.tableIdentifier()
                self.state = 1123
                localctx.kw3 = self.match(sqlParser.ARCHIVE)
                self.state = 1124
                localctx.kw4 = self.match(sqlParser.PARTITION)
                pass
            elif la_ == 35:
                self.enterOuterAlt(localctx, 35)
                self.state = 1126
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1127
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1128
                self.tableIdentifier()
                self.state = 1129
                localctx.kw3 = self.match(sqlParser.UNARCHIVE)
                self.state = 1130
                localctx.kw4 = self.match(sqlParser.PARTITION)
                pass
            elif la_ == 36:
                self.enterOuterAlt(localctx, 36)
                self.state = 1132
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1133
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1134
                self.tableIdentifier()
                self.state = 1135
                localctx.kw3 = self.match(sqlParser.TOUCH)
                pass
            elif la_ == 37:
                self.enterOuterAlt(localctx, 37)
                self.state = 1137
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1138
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1139
                self.tableIdentifier()
                self.state = 1141
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1140
                    self.partitionSpec()
                self.state = 1143
                localctx.kw3 = self.match(sqlParser.COMPACT)
                pass
            elif la_ == 38:
                self.enterOuterAlt(localctx, 38)
                self.state = 1145
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1146
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1147
                self.tableIdentifier()
                self.state = 1149
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1148
                    self.partitionSpec()
                self.state = 1151
                localctx.kw3 = self.match(sqlParser.CONCATENATE)
                pass
            elif la_ == 39:
                self.enterOuterAlt(localctx, 39)
                self.state = 1153
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1154
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1155
                self.tableIdentifier()
                self.state = 1157
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1156
                    self.partitionSpec()
                self.state = 1159
                localctx.kw3 = self.match(sqlParser.SET)
                self.state = 1160
                localctx.kw4 = self.match(sqlParser.FILEFORMAT)
                pass
            elif la_ == 40:
                self.enterOuterAlt(localctx, 40)
                self.state = 1162
                localctx.kw1 = self.match(sqlParser.ALTER)
                self.state = 1163
                localctx.kw2 = self.match(sqlParser.TABLE)
                self.state = 1164
                self.tableIdentifier()
                self.state = 1166
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1165
                    self.partitionSpec()
                self.state = 1168
                localctx.kw3 = self.match(sqlParser.REPLACE)
                self.state = 1169
                localctx.kw4 = self.match(sqlParser.COLUMNS)
                pass
            elif la_ == 41:
                self.enterOuterAlt(localctx, 41)
                self.state = 1171
                localctx.kw1 = self.match(sqlParser.START)
                self.state = 1172
                localctx.kw2 = self.match(sqlParser.TRANSACTION)
                pass
            elif la_ == 42:
                self.enterOuterAlt(localctx, 42)
                self.state = 1173
                localctx.kw1 = self.match(sqlParser.COMMIT)
                pass
            elif la_ == 43:
                self.enterOuterAlt(localctx, 43)
                self.state = 1174
                localctx.kw1 = self.match(sqlParser.ROLLBACK)
                pass
            elif la_ == 44:
                self.enterOuterAlt(localctx, 44)
                self.state = 1175
                localctx.kw1 = self.match(sqlParser.DFS)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CreateTableHeaderContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def EXTERNAL(self):
            return self.getToken(sqlParser.EXTERNAL, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_createTableHeader

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTableHeader'):
                return visitor.visitCreateTableHeader(self)
            else:
                return visitor.visitChildren(self)

    def createTableHeader(self):
        localctx = sqlParser.CreateTableHeaderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_createTableHeader)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1178
            self.match(sqlParser.CREATE)
            self.state = 1180
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 231:
                self.state = 1179
                self.match(sqlParser.TEMPORARY)
            self.state = 1183
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 87:
                self.state = 1182
                self.match(sqlParser.EXTERNAL)
            self.state = 1185
            self.match(sqlParser.TABLE)
            self.state = 1189
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 117, self._ctx)
            if la_ == 1:
                self.state = 1186
                self.match(sqlParser.IF)
                self.state = 1187
                self.match(sqlParser.NOT)
                self.state = 1188
                self.match(sqlParser.EXISTS)
            self.state = 1191
            self.multipartIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ReplaceTableHeaderContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_replaceTableHeader

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitReplaceTableHeader'):
                return visitor.visitReplaceTableHeader(self)
            else:
                return visitor.visitChildren(self)

    def replaceTableHeader(self):
        localctx = sqlParser.ReplaceTableHeaderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_replaceTableHeader)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1195
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 53:
                self.state = 1193
                self.match(sqlParser.CREATE)
                self.state = 1194
                self.match(sqlParser.OR)
            self.state = 1197
            self.match(sqlParser.REPLACE)
            self.state = 1198
            self.match(sqlParser.TABLE)
            self.state = 1199
            self.multipartIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class BucketSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CLUSTERED(self):
            return self.getToken(sqlParser.CLUSTERED, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def INTO(self):
            return self.getToken(sqlParser.INTO, 0)

        def INTEGER_VALUE(self):
            return self.getToken(sqlParser.INTEGER_VALUE, 0)

        def BUCKETS(self):
            return self.getToken(sqlParser.BUCKETS, 0)

        def SORTED(self):
            return self.getToken(sqlParser.SORTED, 0)

        def orderedIdentifierList(self):
            return self.getTypedRuleContext(sqlParser.OrderedIdentifierListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_bucketSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBucketSpec'):
                return visitor.visitBucketSpec(self)
            else:
                return visitor.visitChildren(self)

    def bucketSpec(self):
        localctx = sqlParser.BucketSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_bucketSpec)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1201
            self.match(sqlParser.CLUSTERED)
            self.state = 1202
            self.match(sqlParser.BY)
            self.state = 1203
            self.identifierList()
            self.state = 1207
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 219:
                self.state = 1204
                self.match(sqlParser.SORTED)
                self.state = 1205
                self.match(sqlParser.BY)
                self.state = 1206
                self.orderedIdentifierList()
            self.state = 1209
            self.match(sqlParser.INTO)
            self.state = 1210
            self.match(sqlParser.INTEGER_VALUE)
            self.state = 1211
            self.match(sqlParser.BUCKETS)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SkewSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SKEWED(self):
            return self.getToken(sqlParser.SKEWED, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def constantList(self):
            return self.getTypedRuleContext(sqlParser.ConstantListContext, 0)

        def nestedConstantList(self):
            return self.getTypedRuleContext(sqlParser.NestedConstantListContext, 0)

        def STORED(self):
            return self.getToken(sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def DIRECTORIES(self):
            return self.getToken(sqlParser.DIRECTORIES, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_skewSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSkewSpec'):
                return visitor.visitSkewSpec(self)
            else:
                return visitor.visitChildren(self)

    def skewSpec(self):
        localctx = sqlParser.SkewSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_skewSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1213
            self.match(sqlParser.SKEWED)
            self.state = 1214
            self.match(sqlParser.BY)
            self.state = 1215
            self.identifierList()
            self.state = 1216
            self.match(sqlParser.ON)
            self.state = 1219
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 120, self._ctx)
            if la_ == 1:
                self.state = 1217
                self.constantList()
                pass
            elif la_ == 2:
                self.state = 1218
                self.nestedConstantList()
                pass
            self.state = 1224
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 121, self._ctx)
            if la_ == 1:
                self.state = 1221
                self.match(sqlParser.STORED)
                self.state = 1222
                self.match(sqlParser.AS)
                self.state = 1223
                self.match(sqlParser.DIRECTORIES)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LocationSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LOCATION(self):
            return self.getToken(sqlParser.LOCATION, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_locationSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLocationSpec'):
                return visitor.visitLocationSpec(self)
            else:
                return visitor.visitChildren(self)

    def locationSpec(self):
        localctx = sqlParser.LocationSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_locationSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1226
            self.match(sqlParser.LOCATION)
            self.state = 1227
            self.match(sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CommentSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def COMMENT(self):
            return self.getToken(sqlParser.COMMENT, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_commentSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCommentSpec'):
                return visitor.visitCommentSpec(self)
            else:
                return visitor.visitChildren(self)

    def commentSpec(self):
        localctx = sqlParser.CommentSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_commentSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1229
            self.match(sqlParser.COMMENT)
            self.state = 1230
            self.match(sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QueryContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def queryTerm(self):
            return self.getTypedRuleContext(sqlParser.QueryTermContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(sqlParser.QueryOrganizationContext, 0)

        def ctes(self):
            return self.getTypedRuleContext(sqlParser.CtesContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_query

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQuery'):
                return visitor.visitQuery(self)
            else:
                return visitor.visitChildren(self)

    def query(self):
        localctx = sqlParser.QueryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_query)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1233
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 262:
                self.state = 1232
                self.ctes()
            self.state = 1235
            self.queryTerm(0)
            self.state = 1236
            self.queryOrganization()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class InsertIntoContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_insertInto

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class InsertOverwriteHiveDirContext(InsertIntoContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.path = None
            self.copyFrom(ctx)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def DIRECTORY(self):
            return self.getToken(sqlParser.DIRECTORY, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def LOCAL(self):
            return self.getToken(sqlParser.LOCAL, 0)

        def rowFormat(self):
            return self.getTypedRuleContext(sqlParser.RowFormatContext, 0)

        def createFileFormat(self):
            return self.getTypedRuleContext(sqlParser.CreateFileFormatContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInsertOverwriteHiveDir'):
                return visitor.visitInsertOverwriteHiveDir(self)
            else:
                return visitor.visitChildren(self)

    class InsertOverwriteDirContext(InsertIntoContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.path = None
            self.options = None
            self.copyFrom(ctx)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def DIRECTORY(self):
            return self.getToken(sqlParser.DIRECTORY, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(sqlParser.TableProviderContext, 0)

        def LOCAL(self):
            return self.getToken(sqlParser.LOCAL, 0)

        def OPTIONS(self):
            return self.getToken(sqlParser.OPTIONS, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInsertOverwriteDir'):
                return visitor.visitInsertOverwriteDir(self)
            else:
                return visitor.visitChildren(self)

    class InsertOverwriteTableContext(InsertIntoContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInsertOverwriteTable'):
                return visitor.visitInsertOverwriteTable(self)
            else:
                return visitor.visitChildren(self)

    class InsertIntoTableContext(InsertIntoContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def INTO(self):
            return self.getToken(sqlParser.INTO, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInsertIntoTable'):
                return visitor.visitInsertIntoTable(self)
            else:
                return visitor.visitChildren(self)

    def insertInto(self):
        localctx = sqlParser.InsertIntoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_insertInto)
        self._la = 0
        try:
            self.state = 1293
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 135, self._ctx)
            if la_ == 1:
                localctx = sqlParser.InsertOverwriteTableContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1238
                self.match(sqlParser.INSERT)
                self.state = 1239
                self.match(sqlParser.OVERWRITE)
                self.state = 1241
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 123, self._ctx)
                if la_ == 1:
                    self.state = 1240
                    self.match(sqlParser.TABLE)
                self.state = 1243
                self.multipartIdentifier()
                self.state = 1250
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1244
                    self.partitionSpec()
                    self.state = 1248
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 110:
                        self.state = 1245
                        self.match(sqlParser.IF)
                        self.state = 1246
                        self.match(sqlParser.NOT)
                        self.state = 1247
                        self.match(sqlParser.EXISTS)
                pass
            elif la_ == 2:
                localctx = sqlParser.InsertIntoTableContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1252
                self.match(sqlParser.INSERT)
                self.state = 1253
                self.match(sqlParser.INTO)
                self.state = 1255
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 126, self._ctx)
                if la_ == 1:
                    self.state = 1254
                    self.match(sqlParser.TABLE)
                self.state = 1257
                self.multipartIdentifier()
                self.state = 1259
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 170:
                    self.state = 1258
                    self.partitionSpec()
                self.state = 1264
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 110:
                    self.state = 1261
                    self.match(sqlParser.IF)
                    self.state = 1262
                    self.match(sqlParser.NOT)
                    self.state = 1263
                    self.match(sqlParser.EXISTS)
                pass
            elif la_ == 3:
                localctx = sqlParser.InsertOverwriteHiveDirContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 1266
                self.match(sqlParser.INSERT)
                self.state = 1267
                self.match(sqlParser.OVERWRITE)
                self.state = 1269
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 137:
                    self.state = 1268
                    self.match(sqlParser.LOCAL)
                self.state = 1271
                self.match(sqlParser.DIRECTORY)
                self.state = 1272
                localctx.path = self.match(sqlParser.STRING)
                self.state = 1274
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 202:
                    self.state = 1273
                    self.rowFormat()
                self.state = 1277
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 222:
                    self.state = 1276
                    self.createFileFormat()
                pass
            elif la_ == 4:
                localctx = sqlParser.InsertOverwriteDirContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 1279
                self.match(sqlParser.INSERT)
                self.state = 1280
                self.match(sqlParser.OVERWRITE)
                self.state = 1282
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 137:
                    self.state = 1281
                    self.match(sqlParser.LOCAL)
                self.state = 1284
                self.match(sqlParser.DIRECTORY)
                self.state = 1286
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 283:
                    self.state = 1285
                    localctx.path = self.match(sqlParser.STRING)
                self.state = 1288
                self.tableProvider()
                self.state = 1291
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 160:
                    self.state = 1289
                    self.match(sqlParser.OPTIONS)
                    self.state = 1290
                    localctx.options = self.tablePropertyList()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PartitionSpecLocationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def partitionSpec(self):
            return self.getTypedRuleContext(sqlParser.PartitionSpecContext, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(sqlParser.LocationSpecContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_partitionSpecLocation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionSpecLocation'):
                return visitor.visitPartitionSpecLocation(self)
            else:
                return visitor.visitChildren(self)

    def partitionSpecLocation(self):
        localctx = sqlParser.PartitionSpecLocationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_partitionSpecLocation)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1295
            self.partitionSpec()
            self.state = 1297
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 138:
                self.state = 1296
                self.locationSpec()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PartitionSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PARTITION(self):
            return self.getToken(sqlParser.PARTITION, 0)

        def partitionVal(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PartitionValContext)
            else:
                return self.getTypedRuleContext(sqlParser.PartitionValContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_partitionSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionSpec'):
                return visitor.visitPartitionSpec(self)
            else:
                return visitor.visitChildren(self)

    def partitionSpec(self):
        localctx = sqlParser.PartitionSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_partitionSpec)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1299
            self.match(sqlParser.PARTITION)
            self.state = 1300
            self.match(sqlParser.T__1)
            self.state = 1301
            self.partitionVal()
            self.state = 1306
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1302
                self.match(sqlParser.T__3)
                self.state = 1303
                self.partitionVal()
                self.state = 1308
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1309
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PartitionValContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def EQ(self):
            return self.getToken(sqlParser.EQ, 0)

        def constant(self):
            return self.getTypedRuleContext(sqlParser.ConstantContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_partitionVal

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionVal'):
                return visitor.visitPartitionVal(self)
            else:
                return visitor.visitChildren(self)

    def partitionVal(self):
        localctx = sqlParser.PartitionValContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_partitionVal)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1311
            self.identifier()
            self.state = 1314
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 264:
                self.state = 1312
                self.match(sqlParser.EQ)
                self.state = 1313
                self.constant()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NamespaceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAMESPACE(self):
            return self.getToken(sqlParser.NAMESPACE, 0)

        def DATABASE(self):
            return self.getToken(sqlParser.DATABASE, 0)

        def SCHEMA(self):
            return self.getToken(sqlParser.SCHEMA, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_namespace

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamespace'):
                return visitor.visitNamespace(self)
            else:
                return visitor.visitChildren(self)

    def namespace(self):
        localctx = sqlParser.NamespaceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_namespace)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1316
            _la = self._input.LA(1)
            if not (_la == 62 or _la == 149 or _la == 204):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DescribeFuncNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def comparisonOperator(self):
            return self.getTypedRuleContext(sqlParser.ComparisonOperatorContext, 0)

        def arithmeticOperator(self):
            return self.getTypedRuleContext(sqlParser.ArithmeticOperatorContext, 0)

        def predicateOperator(self):
            return self.getTypedRuleContext(sqlParser.PredicateOperatorContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_describeFuncName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeFuncName'):
                return visitor.visitDescribeFuncName(self)
            else:
                return visitor.visitChildren(self)

    def describeFuncName(self):
        localctx = sqlParser.DescribeFuncNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_describeFuncName)
        try:
            self.state = 1323
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 139, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1318
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1319
                self.match(sqlParser.STRING)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 1320
                self.comparisonOperator()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 1321
                self.arithmeticOperator()
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 1322
                self.predicateOperator()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DescribeColNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._identifier = None
            self.nameParts = list()

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_describeColName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeColName'):
                return visitor.visitDescribeColName(self)
            else:
                return visitor.visitChildren(self)

    def describeColName(self):
        localctx = sqlParser.DescribeColNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_describeColName)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1325
            localctx._identifier = self.identifier()
            localctx.nameParts.append(localctx._identifier)
            self.state = 1330
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 5:
                self.state = 1326
                self.match(sqlParser.T__4)
                self.state = 1327
                localctx._identifier = self.identifier()
                localctx.nameParts.append(localctx._identifier)
                self.state = 1332
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CtesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def namedQuery(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NamedQueryContext)
            else:
                return self.getTypedRuleContext(sqlParser.NamedQueryContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_ctes

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCtes'):
                return visitor.visitCtes(self)
            else:
                return visitor.visitChildren(self)

    def ctes(self):
        localctx = sqlParser.CtesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_ctes)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1333
            self.match(sqlParser.WITH)
            self.state = 1334
            self.namedQuery()
            self.state = 1339
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1335
                self.match(sqlParser.T__3)
                self.state = 1336
                self.namedQuery()
                self.state = 1341
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NamedQueryContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None
            self.columnAliases = None

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_namedQuery

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedQuery'):
                return visitor.visitNamedQuery(self)
            else:
                return visitor.visitChildren(self)

    def namedQuery(self):
        localctx = sqlParser.NamedQueryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_namedQuery)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1342
            localctx.name = self.errorCapturingIdentifier()
            self.state = 1344
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 142, self._ctx)
            if la_ == 1:
                self.state = 1343
                localctx.columnAliases = self.identifierList()
            self.state = 1347
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 22:
                self.state = 1346
                self.match(sqlParser.AS)
            self.state = 1349
            self.match(sqlParser.T__1)
            self.state = 1350
            self.query()
            self.state = 1351
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TableProviderContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_tableProvider

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableProvider'):
                return visitor.visitTableProvider(self)
            else:
                return visitor.visitChildren(self)

    def tableProvider(self):
        localctx = sqlParser.TableProviderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_tableProvider)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1353
            self.match(sqlParser.USING)
            self.state = 1354
            self.multipartIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CreateTableClausesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.options = None
            self.partitioning = None
            self.tableProps = None

        def bucketSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.BucketSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.BucketSpecContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.LocationSpecContext, i)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(sqlParser.CommentSpecContext, i)

        def OPTIONS(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.OPTIONS)
            else:
                return self.getToken(sqlParser.OPTIONS, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.PARTITIONED)
            else:
                return self.getToken(sqlParser.PARTITIONED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyListContext, i)

        def transformList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TransformListContext)
            else:
                return self.getTypedRuleContext(sqlParser.TransformListContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_createTableClauses

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTableClauses'):
                return visitor.visitCreateTableClauses(self)
            else:
                return visitor.visitChildren(self)

    def createTableClauses(self):
        localctx = sqlParser.CreateTableClausesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_createTableClauses)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1368
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 39 or _la == 45 or (_la - 138 & ~63 == 0 and 1 << _la - 138 & 8594128897 != 0) or (_la == 230):
                self.state = 1366
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [160]:
                    self.state = 1356
                    self.match(sqlParser.OPTIONS)
                    self.state = 1357
                    localctx.options = self.tablePropertyList()
                    pass
                elif token in [171]:
                    self.state = 1358
                    self.match(sqlParser.PARTITIONED)
                    self.state = 1359
                    self.match(sqlParser.BY)
                    self.state = 1360
                    localctx.partitioning = self.transformList()
                    pass
                elif token in [39]:
                    self.state = 1361
                    self.bucketSpec()
                    pass
                elif token in [138]:
                    self.state = 1362
                    self.locationSpec()
                    pass
                elif token in [45]:
                    self.state = 1363
                    self.commentSpec()
                    pass
                elif token in [230]:
                    self.state = 1364
                    self.match(sqlParser.TBLPROPERTIES)
                    self.state = 1365
                    localctx.tableProps = self.tablePropertyList()
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 1370
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TablePropertyListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def tableProperty(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TablePropertyContext)
            else:
                return self.getTypedRuleContext(sqlParser.TablePropertyContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_tablePropertyList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyList'):
                return visitor.visitTablePropertyList(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyList(self):
        localctx = sqlParser.TablePropertyListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_tablePropertyList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1371
            self.match(sqlParser.T__1)
            self.state = 1372
            self.tableProperty()
            self.state = 1377
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1373
                self.match(sqlParser.T__3)
                self.state = 1374
                self.tableProperty()
                self.state = 1379
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1380
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TablePropertyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def tablePropertyKey(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyKeyContext, 0)

        def tablePropertyValue(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyValueContext, 0)

        def EQ(self):
            return self.getToken(sqlParser.EQ, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_tableProperty

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableProperty'):
                return visitor.visitTableProperty(self)
            else:
                return visitor.visitChildren(self)

    def tableProperty(self):
        localctx = sqlParser.TablePropertyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_tableProperty)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1382
            localctx.key = self.tablePropertyKey()
            self.state = 1387
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 89 or (_la - 241 & ~63 == 0 and 1 << _la - 241 & 356241775788033 != 0):
                self.state = 1384
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 264:
                    self.state = 1383
                    self.match(sqlParser.EQ)
                self.state = 1386
                localctx.value = self.tablePropertyValue()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TablePropertyKeyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_tablePropertyKey

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyKey'):
                return visitor.visitTablePropertyKey(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyKey(self):
        localctx = sqlParser.TablePropertyKeyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_tablePropertyKey)
        self._la = 0
        try:
            self.state = 1398
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 150, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1389
                self.identifier()
                self.state = 1394
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 5:
                    self.state = 1390
                    self.match(sqlParser.T__4)
                    self.state = 1391
                    self.identifier()
                    self.state = 1396
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1397
                self.match(sqlParser.STRING)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TablePropertyValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INTEGER_VALUE(self):
            return self.getToken(sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(sqlParser.DECIMAL_VALUE, 0)

        def booleanValue(self):
            return self.getTypedRuleContext(sqlParser.BooleanValueContext, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_tablePropertyValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyValue'):
                return visitor.visitTablePropertyValue(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyValue(self):
        localctx = sqlParser.TablePropertyValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_tablePropertyValue)
        try:
            self.state = 1404
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [287]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1400
                self.match(sqlParser.INTEGER_VALUE)
                pass
            elif token in [289]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1401
                self.match(sqlParser.DECIMAL_VALUE)
                pass
            elif token in [89, 241]:
                self.enterOuterAlt(localctx, 3)
                self.state = 1402
                self.booleanValue()
                pass
            elif token in [283]:
                self.enterOuterAlt(localctx, 4)
                self.state = 1403
                self.match(sqlParser.STRING)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ConstantListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constant(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ConstantContext)
            else:
                return self.getTypedRuleContext(sqlParser.ConstantContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_constantList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitConstantList'):
                return visitor.visitConstantList(self)
            else:
                return visitor.visitChildren(self)

    def constantList(self):
        localctx = sqlParser.ConstantListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_constantList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1406
            self.match(sqlParser.T__1)
            self.state = 1407
            self.constant()
            self.state = 1412
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1408
                self.match(sqlParser.T__3)
                self.state = 1409
                self.constant()
                self.state = 1414
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1415
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NestedConstantListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def constantList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ConstantListContext)
            else:
                return self.getTypedRuleContext(sqlParser.ConstantListContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_nestedConstantList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNestedConstantList'):
                return visitor.visitNestedConstantList(self)
            else:
                return visitor.visitChildren(self)

    def nestedConstantList(self):
        localctx = sqlParser.NestedConstantListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 64, self.RULE_nestedConstantList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1417
            self.match(sqlParser.T__1)
            self.state = 1418
            self.constantList()
            self.state = 1423
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1419
                self.match(sqlParser.T__3)
                self.state = 1420
                self.constantList()
                self.state = 1425
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1426
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class CreateFileFormatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STORED(self):
            return self.getToken(sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def fileFormat(self):
            return self.getTypedRuleContext(sqlParser.FileFormatContext, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def storageHandler(self):
            return self.getTypedRuleContext(sqlParser.StorageHandlerContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_createFileFormat

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateFileFormat'):
                return visitor.visitCreateFileFormat(self)
            else:
                return visitor.visitChildren(self)

    def createFileFormat(self):
        localctx = sqlParser.CreateFileFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_createFileFormat)
        try:
            self.state = 1434
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 154, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1428
                self.match(sqlParser.STORED)
                self.state = 1429
                self.match(sqlParser.AS)
                self.state = 1430
                self.fileFormat()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1431
                self.match(sqlParser.STORED)
                self.state = 1432
                self.match(sqlParser.BY)
                self.state = 1433
                self.storageHandler()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FileFormatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_fileFormat

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class TableFileFormatContext(FileFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.inFmt = None
            self.outFmt = None
            self.copyFrom(ctx)

        def INPUTFORMAT(self):
            return self.getToken(sqlParser.INPUTFORMAT, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(sqlParser.OUTPUTFORMAT, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.STRING)
            else:
                return self.getToken(sqlParser.STRING, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableFileFormat'):
                return visitor.visitTableFileFormat(self)
            else:
                return visitor.visitChildren(self)

    class GenericFileFormatContext(FileFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitGenericFileFormat'):
                return visitor.visitGenericFileFormat(self)
            else:
                return visitor.visitChildren(self)

    def fileFormat(self):
        localctx = sqlParser.FileFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_fileFormat)
        try:
            self.state = 1441
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 155, self._ctx)
            if la_ == 1:
                localctx = sqlParser.TableFileFormatContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1436
                self.match(sqlParser.INPUTFORMAT)
                self.state = 1437
                localctx.inFmt = self.match(sqlParser.STRING)
                self.state = 1438
                self.match(sqlParser.OUTPUTFORMAT)
                self.state = 1439
                localctx.outFmt = self.match(sqlParser.STRING)
                pass
            elif la_ == 2:
                localctx = sqlParser.GenericFileFormatContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1440
                self.identifier()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StorageHandlerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_storageHandler

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStorageHandler'):
                return visitor.visitStorageHandler(self)
            else:
                return visitor.visitChildren(self)

    def storageHandler(self):
        localctx = sqlParser.StorageHandlerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 70, self.RULE_storageHandler)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1443
            self.match(sqlParser.STRING)
            self.state = 1447
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 156, self._ctx)
            if la_ == 1:
                self.state = 1444
                self.match(sqlParser.WITH)
                self.state = 1445
                self.match(sqlParser.SERDEPROPERTIES)
                self.state = 1446
                self.tablePropertyList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ResourceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_resource

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitResource'):
                return visitor.visitResource(self)
            else:
                return visitor.visitChildren(self)

    def resource(self):
        localctx = sqlParser.ResourceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 72, self.RULE_resource)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1449
            self.identifier()
            self.state = 1450
            self.match(sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DmlStatementNoWithContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_dmlStatementNoWith

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class DeleteFromTableContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DELETE(self):
            return self.getToken(sqlParser.DELETE, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(sqlParser.WhereClauseContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDeleteFromTable'):
                return visitor.visitDeleteFromTable(self)
            else:
                return visitor.visitChildren(self)

    class SingleInsertQueryContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def insertInto(self):
            return self.getTypedRuleContext(sqlParser.InsertIntoContext, 0)

        def queryTerm(self):
            return self.getTypedRuleContext(sqlParser.QueryTermContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(sqlParser.QueryOrganizationContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleInsertQuery'):
                return visitor.visitSingleInsertQuery(self)
            else:
                return visitor.visitChildren(self)

    class MultiInsertQueryContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fromClause(self):
            return self.getTypedRuleContext(sqlParser.FromClauseContext, 0)

        def multiInsertQueryBody(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultiInsertQueryBodyContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultiInsertQueryBodyContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultiInsertQuery'):
                return visitor.visitMultiInsertQuery(self)
            else:
                return visitor.visitChildren(self)

    class UpdateTableContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def UPDATE(self):
            return self.getToken(sqlParser.UPDATE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def setClause(self):
            return self.getTypedRuleContext(sqlParser.SetClauseContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(sqlParser.WhereClauseContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUpdateTable'):
                return visitor.visitUpdateTable(self)
            else:
                return visitor.visitChildren(self)

    class MergeIntoTableContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.target = None
            self.targetAlias = None
            self.source = None
            self.sourceQuery = None
            self.sourceAlias = None
            self.mergeCondition = None
            self.copyFrom(ctx)

        def MERGE(self):
            return self.getToken(sqlParser.MERGE, 0)

        def INTO(self):
            return self.getToken(sqlParser.INTO, 0)

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def tableAlias(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TableAliasContext)
            else:
                return self.getTypedRuleContext(sqlParser.TableAliasContext, i)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def matchedClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MatchedClauseContext)
            else:
                return self.getTypedRuleContext(sqlParser.MatchedClauseContext, i)

        def notMatchedClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NotMatchedClauseContext)
            else:
                return self.getTypedRuleContext(sqlParser.NotMatchedClauseContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMergeIntoTable'):
                return visitor.visitMergeIntoTable(self)
            else:
                return visitor.visitChildren(self)

    def dmlStatementNoWith(self):
        localctx = sqlParser.DmlStatementNoWithContext(self, self._ctx, self.state)
        self.enterRule(localctx, 74, self.RULE_dmlStatementNoWith)
        self._la = 0
        try:
            self.state = 1503
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [119]:
                localctx = sqlParser.SingleInsertQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1452
                self.insertInto()
                self.state = 1453
                self.queryTerm(0)
                self.state = 1454
                self.queryOrganization()
                pass
            elif token in [100]:
                localctx = sqlParser.MultiInsertQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1456
                self.fromClause()
                self.state = 1458
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 1457
                    self.multiInsertQueryBody()
                    self.state = 1460
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 119:
                        break
                pass
            elif token in [67]:
                localctx = sqlParser.DeleteFromTableContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 1462
                self.match(sqlParser.DELETE)
                self.state = 1463
                self.match(sqlParser.FROM)
                self.state = 1464
                self.multipartIdentifier()
                self.state = 1465
                self.tableAlias()
                self.state = 1467
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 260:
                    self.state = 1466
                    self.whereClause()
                pass
            elif token in [252]:
                localctx = sqlParser.UpdateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 1469
                self.match(sqlParser.UPDATE)
                self.state = 1470
                self.multipartIdentifier()
                self.state = 1471
                self.tableAlias()
                self.state = 1472
                self.setClause()
                self.state = 1474
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 260:
                    self.state = 1473
                    self.whereClause()
                pass
            elif token in [145]:
                localctx = sqlParser.MergeIntoTableContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 1476
                self.match(sqlParser.MERGE)
                self.state = 1477
                self.match(sqlParser.INTO)
                self.state = 1478
                localctx.target = self.multipartIdentifier()
                self.state = 1479
                localctx.targetAlias = self.tableAlias()
                self.state = 1480
                self.match(sqlParser.USING)
                self.state = 1486
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 160, self._ctx)
                if la_ == 1:
                    self.state = 1481
                    localctx.source = self.multipartIdentifier()
                    pass
                elif la_ == 2:
                    self.state = 1482
                    self.match(sqlParser.T__1)
                    self.state = 1483
                    localctx.sourceQuery = self.query()
                    self.state = 1484
                    self.match(sqlParser.T__2)
                    pass
                self.state = 1488
                localctx.sourceAlias = self.tableAlias()
                self.state = 1489
                self.match(sqlParser.ON)
                self.state = 1490
                localctx.mergeCondition = self.booleanExpression(0)
                self.state = 1494
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 161, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1491
                        self.matchedClause()
                    self.state = 1496
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 161, self._ctx)
                self.state = 1500
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 259:
                    self.state = 1497
                    self.notMatchedClause()
                    self.state = 1502
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QueryOrganizationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._sortItem = None
            self.order = list()
            self._expression = None
            self.clusterBy = list()
            self.distributeBy = list()
            self.sort = list()
            self.limit = None

        def ORDER(self):
            return self.getToken(sqlParser.ORDER, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def CLUSTER(self):
            return self.getToken(sqlParser.CLUSTER, 0)

        def DISTRIBUTE(self):
            return self.getToken(sqlParser.DISTRIBUTE, 0)

        def SORT(self):
            return self.getToken(sqlParser.SORT, 0)

        def windowClause(self):
            return self.getTypedRuleContext(sqlParser.WindowClauseContext, 0)

        def LIMIT(self):
            return self.getToken(sqlParser.LIMIT, 0)

        def sortItem(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.SortItemContext)
            else:
                return self.getTypedRuleContext(sqlParser.SortItemContext, i)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def ALL(self):
            return self.getToken(sqlParser.ALL, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_queryOrganization

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQueryOrganization'):
                return visitor.visitQueryOrganization(self)
            else:
                return visitor.visitChildren(self)

    def queryOrganization(self):
        localctx = sqlParser.QueryOrganizationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 76, self.RULE_queryOrganization)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1515
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 165, self._ctx)
            if la_ == 1:
                self.state = 1505
                self.match(sqlParser.ORDER)
                self.state = 1506
                self.match(sqlParser.BY)
                self.state = 1507
                localctx._sortItem = self.sortItem()
                localctx.order.append(localctx._sortItem)
                self.state = 1512
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 164, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1508
                        self.match(sqlParser.T__3)
                        self.state = 1509
                        localctx._sortItem = self.sortItem()
                        localctx.order.append(localctx._sortItem)
                    self.state = 1514
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 164, self._ctx)
            self.state = 1527
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 167, self._ctx)
            if la_ == 1:
                self.state = 1517
                self.match(sqlParser.CLUSTER)
                self.state = 1518
                self.match(sqlParser.BY)
                self.state = 1519
                localctx._expression = self.expression()
                localctx.clusterBy.append(localctx._expression)
                self.state = 1524
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 166, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1520
                        self.match(sqlParser.T__3)
                        self.state = 1521
                        localctx._expression = self.expression()
                        localctx.clusterBy.append(localctx._expression)
                    self.state = 1526
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 166, self._ctx)
            self.state = 1539
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 169, self._ctx)
            if la_ == 1:
                self.state = 1529
                self.match(sqlParser.DISTRIBUTE)
                self.state = 1530
                self.match(sqlParser.BY)
                self.state = 1531
                localctx._expression = self.expression()
                localctx.distributeBy.append(localctx._expression)
                self.state = 1536
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 168, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1532
                        self.match(sqlParser.T__3)
                        self.state = 1533
                        localctx._expression = self.expression()
                        localctx.distributeBy.append(localctx._expression)
                    self.state = 1538
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 168, self._ctx)
            self.state = 1551
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 171, self._ctx)
            if la_ == 1:
                self.state = 1541
                self.match(sqlParser.SORT)
                self.state = 1542
                self.match(sqlParser.BY)
                self.state = 1543
                localctx._sortItem = self.sortItem()
                localctx.sort.append(localctx._sortItem)
                self.state = 1548
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 170, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1544
                        self.match(sqlParser.T__3)
                        self.state = 1545
                        localctx._sortItem = self.sortItem()
                        localctx.sort.append(localctx._sortItem)
                    self.state = 1550
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 170, self._ctx)
            self.state = 1554
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 172, self._ctx)
            if la_ == 1:
                self.state = 1553
                self.windowClause()
            self.state = 1561
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 174, self._ctx)
            if la_ == 1:
                self.state = 1556
                self.match(sqlParser.LIMIT)
                self.state = 1559
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 173, self._ctx)
                if la_ == 1:
                    self.state = 1557
                    self.match(sqlParser.ALL)
                    pass
                elif la_ == 2:
                    self.state = 1558
                    localctx.limit = self.expression()
                    pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MultiInsertQueryBodyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def insertInto(self):
            return self.getTypedRuleContext(sqlParser.InsertIntoContext, 0)

        def fromStatementBody(self):
            return self.getTypedRuleContext(sqlParser.FromStatementBodyContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_multiInsertQueryBody

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultiInsertQueryBody'):
                return visitor.visitMultiInsertQueryBody(self)
            else:
                return visitor.visitChildren(self)

    def multiInsertQueryBody(self):
        localctx = sqlParser.MultiInsertQueryBodyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 78, self.RULE_multiInsertQueryBody)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1563
            self.insertInto()
            self.state = 1564
            self.fromStatementBody()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QueryTermContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_queryTerm

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class QueryTermDefaultContext(QueryTermContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def queryPrimary(self):
            return self.getTypedRuleContext(sqlParser.QueryPrimaryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQueryTermDefault'):
                return visitor.visitQueryTermDefault(self)
            else:
                return visitor.visitChildren(self)

    class SetOperationContext(QueryTermContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.operator = None
            self.right = None
            self.copyFrom(ctx)

        def queryTerm(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.QueryTermContext)
            else:
                return self.getTypedRuleContext(sqlParser.QueryTermContext, i)

        def INTERSECT(self):
            return self.getToken(sqlParser.INTERSECT, 0)

        def UNION(self):
            return self.getToken(sqlParser.UNION, 0)

        def EXCEPT(self):
            return self.getToken(sqlParser.EXCEPT, 0)

        def SETMINUS(self):
            return self.getToken(sqlParser.SETMINUS, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(sqlParser.SetQuantifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetOperation'):
                return visitor.visitSetOperation(self)
            else:
                return visitor.visitChildren(self)

    def queryTerm(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = sqlParser.QueryTermContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 80
        self.enterRecursionRule(localctx, 80, self.RULE_queryTerm, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            localctx = sqlParser.QueryTermDefaultContext(self, localctx)
            self._ctx = localctx
            _prevctx = localctx
            self.state = 1567
            self.queryPrimary()
            self._ctx.stop = self._input.LT(-1)
            self.state = 1592
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 179, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 1590
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 178, self._ctx)
                    if la_ == 1:
                        localctx = sqlParser.SetOperationContext(self, sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 1569
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 3)')
                        self.state = 1570
                        if not self.legacy_setops_precedence_enbled:
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.legacy_setops_precedence_enbled')
                        self.state = 1571
                        localctx.operator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la == 81 or _la == 120 or _la == 213 or (_la == 247)):
                            localctx.operator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 1573
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 14 or _la == 74:
                            self.state = 1572
                            self.setQuantifier()
                        self.state = 1575
                        localctx.right = self.queryTerm(4)
                        pass
                    elif la_ == 2:
                        localctx = sqlParser.SetOperationContext(self, sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 1576
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 1577
                        if not not self.legacy_setops_precedence_enbled:
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'not self.legacy_setops_precedence_enbled')
                        self.state = 1578
                        localctx.operator = self.match(sqlParser.INTERSECT)
                        self.state = 1580
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 14 or _la == 74:
                            self.state = 1579
                            self.setQuantifier()
                        self.state = 1582
                        localctx.right = self.queryTerm(3)
                        pass
                    elif la_ == 3:
                        localctx = sqlParser.SetOperationContext(self, sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 1583
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 1584
                        if not not self.legacy_setops_precedence_enbled:
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'not self.legacy_setops_precedence_enbled')
                        self.state = 1585
                        localctx.operator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la == 81 or _la == 213 or _la == 247):
                            localctx.operator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 1587
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 14 or _la == 74:
                            self.state = 1586
                            self.setQuantifier()
                        self.state = 1589
                        localctx.right = self.queryTerm(2)
                        pass
                self.state = 1594
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 179, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class QueryPrimaryContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_queryPrimary

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class SubqueryContext(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSubquery'):
                return visitor.visitSubquery(self)
            else:
                return visitor.visitChildren(self)

    class QueryPrimaryDefaultContext(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def querySpecification(self):
            return self.getTypedRuleContext(sqlParser.QuerySpecificationContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQueryPrimaryDefault'):
                return visitor.visitQueryPrimaryDefault(self)
            else:
                return visitor.visitChildren(self)

    class InlineTableDefault1Context(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def inlineTable(self):
            return self.getTypedRuleContext(sqlParser.InlineTableContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInlineTableDefault1'):
                return visitor.visitInlineTableDefault1(self)
            else:
                return visitor.visitChildren(self)

    class FromStmtContext(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fromStatement(self):
            return self.getTypedRuleContext(sqlParser.FromStatementContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromStmt'):
                return visitor.visitFromStmt(self)
            else:
                return visitor.visitChildren(self)

    class TableContext(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTable'):
                return visitor.visitTable(self)
            else:
                return visitor.visitChildren(self)

    def queryPrimary(self):
        localctx = sqlParser.QueryPrimaryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 82, self.RULE_queryPrimary)
        try:
            self.state = 1604
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [143, 187, 206]:
                localctx = sqlParser.QueryPrimaryDefaultContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1595
                self.querySpecification()
                pass
            elif token in [100]:
                localctx = sqlParser.FromStmtContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1596
                self.fromStatement()
                pass
            elif token in [227]:
                localctx = sqlParser.TableContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 1597
                self.match(sqlParser.TABLE)
                self.state = 1598
                self.multipartIdentifier()
                pass
            elif token in [256]:
                localctx = sqlParser.InlineTableDefault1Context(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 1599
                self.inlineTable()
                pass
            elif token in [2]:
                localctx = sqlParser.SubqueryContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 1600
                self.match(sqlParser.T__1)
                self.state = 1601
                self.query()
                self.state = 1602
                self.match(sqlParser.T__2)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SortItemContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.ordering = None
            self.nullOrder = None

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def NULLS(self):
            return self.getToken(sqlParser.NULLS, 0)

        def ASC(self):
            return self.getToken(sqlParser.ASC, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def LAST(self):
            return self.getToken(sqlParser.LAST, 0)

        def FIRST(self):
            return self.getToken(sqlParser.FIRST, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_sortItem

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSortItem'):
                return visitor.visitSortItem(self)
            else:
                return visitor.visitChildren(self)

    def sortItem(self):
        localctx = sqlParser.SortItemContext(self, self._ctx, self.state)
        self.enterRule(localctx, 84, self.RULE_sortItem)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1606
            self.expression()
            self.state = 1608
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 181, self._ctx)
            if la_ == 1:
                self.state = 1607
                localctx.ordering = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 23 or _la == 69):
                    localctx.ordering = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
            self.state = 1612
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 182, self._ctx)
            if la_ == 1:
                self.state = 1610
                self.match(sqlParser.NULLS)
                self.state = 1611
                localctx.nullOrder = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 94 or _la == 127):
                    localctx.nullOrder = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FromStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fromClause(self):
            return self.getTypedRuleContext(sqlParser.FromClauseContext, 0)

        def fromStatementBody(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.FromStatementBodyContext)
            else:
                return self.getTypedRuleContext(sqlParser.FromStatementBodyContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_fromStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromStatement'):
                return visitor.visitFromStatement(self)
            else:
                return visitor.visitChildren(self)

    def fromStatement(self):
        localctx = sqlParser.FromStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 86, self.RULE_fromStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1614
            self.fromClause()
            self.state = 1616
            self._errHandler.sync(self)
            _alt = 1
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1615
                    self.fromStatementBody()
                else:
                    raise NoViableAltException(self)
                self.state = 1618
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 183, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FromStatementBodyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def transformClause(self):
            return self.getTypedRuleContext(sqlParser.TransformClauseContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(sqlParser.QueryOrganizationContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(sqlParser.WhereClauseContext, 0)

        def selectClause(self):
            return self.getTypedRuleContext(sqlParser.SelectClauseContext, 0)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(sqlParser.LateralViewContext, i)

        def aggregationClause(self):
            return self.getTypedRuleContext(sqlParser.AggregationClauseContext, 0)

        def havingClause(self):
            return self.getTypedRuleContext(sqlParser.HavingClauseContext, 0)

        def windowClause(self):
            return self.getTypedRuleContext(sqlParser.WindowClauseContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_fromStatementBody

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromStatementBody'):
                return visitor.visitFromStatementBody(self)
            else:
                return visitor.visitChildren(self)

    def fromStatementBody(self):
        localctx = sqlParser.FromStatementBodyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 88, self.RULE_fromStatementBody)
        try:
            self.state = 1647
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 190, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1620
                self.transformClause()
                self.state = 1622
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 184, self._ctx)
                if la_ == 1:
                    self.state = 1621
                    self.whereClause()
                self.state = 1624
                self.queryOrganization()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1626
                self.selectClause()
                self.state = 1630
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 185, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1627
                        self.lateralView()
                    self.state = 1632
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 185, self._ctx)
                self.state = 1634
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 186, self._ctx)
                if la_ == 1:
                    self.state = 1633
                    self.whereClause()
                self.state = 1637
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 187, self._ctx)
                if la_ == 1:
                    self.state = 1636
                    self.aggregationClause()
                self.state = 1640
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 188, self._ctx)
                if la_ == 1:
                    self.state = 1639
                    self.havingClause()
                self.state = 1643
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 189, self._ctx)
                if la_ == 1:
                    self.state = 1642
                    self.windowClause()
                self.state = 1645
                self.queryOrganization()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QuerySpecificationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_querySpecification

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class RegularQuerySpecificationContext(QuerySpecificationContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def selectClause(self):
            return self.getTypedRuleContext(sqlParser.SelectClauseContext, 0)

        def fromClause(self):
            return self.getTypedRuleContext(sqlParser.FromClauseContext, 0)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(sqlParser.LateralViewContext, i)

        def whereClause(self):
            return self.getTypedRuleContext(sqlParser.WhereClauseContext, 0)

        def aggregationClause(self):
            return self.getTypedRuleContext(sqlParser.AggregationClauseContext, 0)

        def havingClause(self):
            return self.getTypedRuleContext(sqlParser.HavingClauseContext, 0)

        def windowClause(self):
            return self.getTypedRuleContext(sqlParser.WindowClauseContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRegularQuerySpecification'):
                return visitor.visitRegularQuerySpecification(self)
            else:
                return visitor.visitChildren(self)

    class TransformQuerySpecificationContext(QuerySpecificationContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def transformClause(self):
            return self.getTypedRuleContext(sqlParser.TransformClauseContext, 0)

        def fromClause(self):
            return self.getTypedRuleContext(sqlParser.FromClauseContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(sqlParser.WhereClauseContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformQuerySpecification'):
                return visitor.visitTransformQuerySpecification(self)
            else:
                return visitor.visitChildren(self)

    def querySpecification(self):
        localctx = sqlParser.QuerySpecificationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 90, self.RULE_querySpecification)
        try:
            self.state = 1678
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 199, self._ctx)
            if la_ == 1:
                localctx = sqlParser.TransformQuerySpecificationContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1649
                self.transformClause()
                self.state = 1651
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 191, self._ctx)
                if la_ == 1:
                    self.state = 1650
                    self.fromClause()
                self.state = 1654
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 192, self._ctx)
                if la_ == 1:
                    self.state = 1653
                    self.whereClause()
                pass
            elif la_ == 2:
                localctx = sqlParser.RegularQuerySpecificationContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1656
                self.selectClause()
                self.state = 1658
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 193, self._ctx)
                if la_ == 1:
                    self.state = 1657
                    self.fromClause()
                self.state = 1663
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 194, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1660
                        self.lateralView()
                    self.state = 1665
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 194, self._ctx)
                self.state = 1667
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 195, self._ctx)
                if la_ == 1:
                    self.state = 1666
                    self.whereClause()
                self.state = 1670
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 196, self._ctx)
                if la_ == 1:
                    self.state = 1669
                    self.aggregationClause()
                self.state = 1673
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 197, self._ctx)
                if la_ == 1:
                    self.state = 1672
                    self.havingClause()
                self.state = 1676
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 198, self._ctx)
                if la_ == 1:
                    self.state = 1675
                    self.windowClause()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransformClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.kind = None
            self.inRowFormat = None
            self.recordWriter = None
            self.script = None
            self.outRowFormat = None
            self.recordReader = None

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.STRING)
            else:
                return self.getToken(sqlParser.STRING, i)

        def SELECT(self):
            return self.getToken(sqlParser.SELECT, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(sqlParser.NamedExpressionSeqContext, 0)

        def TRANSFORM(self):
            return self.getToken(sqlParser.TRANSFORM, 0)

        def MAP(self):
            return self.getToken(sqlParser.MAP, 0)

        def REDUCE(self):
            return self.getToken(sqlParser.REDUCE, 0)

        def RECORDWRITER(self):
            return self.getToken(sqlParser.RECORDWRITER, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def RECORDREADER(self):
            return self.getToken(sqlParser.RECORDREADER, 0)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(sqlParser.RowFormatContext, i)

        def identifierSeq(self):
            return self.getTypedRuleContext(sqlParser.IdentifierSeqContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(sqlParser.ColTypeListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_transformClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformClause'):
                return visitor.visitTransformClause(self)
            else:
                return visitor.visitChildren(self)

    def transformClause(self):
        localctx = sqlParser.TransformClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 92, self.RULE_transformClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1690
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [206]:
                self.state = 1680
                self.match(sqlParser.SELECT)
                self.state = 1681
                localctx.kind = self.match(sqlParser.TRANSFORM)
                self.state = 1682
                self.match(sqlParser.T__1)
                self.state = 1683
                self.namedExpressionSeq()
                self.state = 1684
                self.match(sqlParser.T__2)
                pass
            elif token in [143]:
                self.state = 1686
                localctx.kind = self.match(sqlParser.MAP)
                self.state = 1687
                self.namedExpressionSeq()
                pass
            elif token in [187]:
                self.state = 1688
                localctx.kind = self.match(sqlParser.REDUCE)
                self.state = 1689
                self.namedExpressionSeq()
                pass
            else:
                raise NoViableAltException(self)
            self.state = 1693
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 202:
                self.state = 1692
                localctx.inRowFormat = self.rowFormat()
            self.state = 1697
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 185:
                self.state = 1695
                self.match(sqlParser.RECORDWRITER)
                self.state = 1696
                localctx.recordWriter = self.match(sqlParser.STRING)
            self.state = 1699
            self.match(sqlParser.USING)
            self.state = 1700
            localctx.script = self.match(sqlParser.STRING)
            self.state = 1713
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 205, self._ctx)
            if la_ == 1:
                self.state = 1701
                self.match(sqlParser.AS)
                self.state = 1711
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 204, self._ctx)
                if la_ == 1:
                    self.state = 1702
                    self.identifierSeq()
                    pass
                elif la_ == 2:
                    self.state = 1703
                    self.colTypeList()
                    pass
                elif la_ == 3:
                    self.state = 1704
                    self.match(sqlParser.T__1)
                    self.state = 1707
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 203, self._ctx)
                    if la_ == 1:
                        self.state = 1705
                        self.identifierSeq()
                        pass
                    elif la_ == 2:
                        self.state = 1706
                        self.colTypeList()
                        pass
                    self.state = 1709
                    self.match(sqlParser.T__2)
                    pass
            self.state = 1716
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 206, self._ctx)
            if la_ == 1:
                self.state = 1715
                localctx.outRowFormat = self.rowFormat()
            self.state = 1720
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 207, self._ctx)
            if la_ == 1:
                self.state = 1718
                self.match(sqlParser.RECORDREADER)
                self.state = 1719
                localctx.recordReader = self.match(sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SelectClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._hint = None
            self.hints = list()

        def SELECT(self):
            return self.getToken(sqlParser.SELECT, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(sqlParser.NamedExpressionSeqContext, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(sqlParser.SetQuantifierContext, 0)

        def hint(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.HintContext)
            else:
                return self.getTypedRuleContext(sqlParser.HintContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_selectClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSelectClause'):
                return visitor.visitSelectClause(self)
            else:
                return visitor.visitChildren(self)

    def selectClause(self):
        localctx = sqlParser.SelectClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 94, self.RULE_selectClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1722
            self.match(sqlParser.SELECT)
            self.state = 1726
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 208, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1723
                    localctx._hint = self.hint()
                    localctx.hints.append(localctx._hint)
                self.state = 1728
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 208, self._ctx)
            self.state = 1730
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 209, self._ctx)
            if la_ == 1:
                self.state = 1729
                self.setQuantifier()
            self.state = 1732
            self.namedExpressionSeq()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SetClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def assignmentList(self):
            return self.getTypedRuleContext(sqlParser.AssignmentListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_setClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetClause'):
                return visitor.visitSetClause(self)
            else:
                return visitor.visitChildren(self)

    def setClause(self):
        localctx = sqlParser.SetClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 96, self.RULE_setClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1734
            self.match(sqlParser.SET)
            self.state = 1735
            self.assignmentList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MatchedClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.matchedCond = None

        def WHEN(self):
            return self.getToken(sqlParser.WHEN, 0)

        def MATCHED(self):
            return self.getToken(sqlParser.MATCHED, 0)

        def THEN(self):
            return self.getToken(sqlParser.THEN, 0)

        def matchedAction(self):
            return self.getTypedRuleContext(sqlParser.MatchedActionContext, 0)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_matchedClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMatchedClause'):
                return visitor.visitMatchedClause(self)
            else:
                return visitor.visitChildren(self)

    def matchedClause(self):
        localctx = sqlParser.MatchedClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 98, self.RULE_matchedClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1737
            self.match(sqlParser.WHEN)
            self.state = 1738
            self.match(sqlParser.MATCHED)
            self.state = 1741
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 17:
                self.state = 1739
                self.match(sqlParser.AND)
                self.state = 1740
                localctx.matchedCond = self.booleanExpression(0)
            self.state = 1743
            self.match(sqlParser.THEN)
            self.state = 1744
            self.matchedAction()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NotMatchedClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.notMatchedCond = None

        def WHEN(self):
            return self.getToken(sqlParser.WHEN, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def MATCHED(self):
            return self.getToken(sqlParser.MATCHED, 0)

        def THEN(self):
            return self.getToken(sqlParser.THEN, 0)

        def notMatchedAction(self):
            return self.getTypedRuleContext(sqlParser.NotMatchedActionContext, 0)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_notMatchedClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNotMatchedClause'):
                return visitor.visitNotMatchedClause(self)
            else:
                return visitor.visitChildren(self)

    def notMatchedClause(self):
        localctx = sqlParser.NotMatchedClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 100, self.RULE_notMatchedClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1746
            self.match(sqlParser.WHEN)
            self.state = 1747
            self.match(sqlParser.NOT)
            self.state = 1748
            self.match(sqlParser.MATCHED)
            self.state = 1751
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 17:
                self.state = 1749
                self.match(sqlParser.AND)
                self.state = 1750
                localctx.notMatchedCond = self.booleanExpression(0)
            self.state = 1753
            self.match(sqlParser.THEN)
            self.state = 1754
            self.notMatchedAction()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MatchedActionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DELETE(self):
            return self.getToken(sqlParser.DELETE, 0)

        def UPDATE(self):
            return self.getToken(sqlParser.UPDATE, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def ASTERISK(self):
            return self.getToken(sqlParser.ASTERISK, 0)

        def assignmentList(self):
            return self.getTypedRuleContext(sqlParser.AssignmentListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_matchedAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMatchedAction'):
                return visitor.visitMatchedAction(self)
            else:
                return visitor.visitChildren(self)

    def matchedAction(self):
        localctx = sqlParser.MatchedActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 102, self.RULE_matchedAction)
        try:
            self.state = 1763
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 212, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1756
                self.match(sqlParser.DELETE)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1757
                self.match(sqlParser.UPDATE)
                self.state = 1758
                self.match(sqlParser.SET)
                self.state = 1759
                self.match(sqlParser.ASTERISK)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 1760
                self.match(sqlParser.UPDATE)
                self.state = 1761
                self.match(sqlParser.SET)
                self.state = 1762
                self.assignmentList()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NotMatchedActionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.columns = None

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def ASTERISK(self):
            return self.getToken(sqlParser.ASTERISK, 0)

        def VALUES(self):
            return self.getToken(sqlParser.VALUES, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def multipartIdentifierList(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_notMatchedAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNotMatchedAction'):
                return visitor.visitNotMatchedAction(self)
            else:
                return visitor.visitChildren(self)

    def notMatchedAction(self):
        localctx = sqlParser.NotMatchedActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 104, self.RULE_notMatchedAction)
        self._la = 0
        try:
            self.state = 1783
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 214, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1765
                self.match(sqlParser.INSERT)
                self.state = 1766
                self.match(sqlParser.ASTERISK)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1767
                self.match(sqlParser.INSERT)
                self.state = 1768
                self.match(sqlParser.T__1)
                self.state = 1769
                localctx.columns = self.multipartIdentifierList()
                self.state = 1770
                self.match(sqlParser.T__2)
                self.state = 1771
                self.match(sqlParser.VALUES)
                self.state = 1772
                self.match(sqlParser.T__1)
                self.state = 1773
                self.expression()
                self.state = 1778
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 1774
                    self.match(sqlParser.T__3)
                    self.state = 1775
                    self.expression()
                    self.state = 1780
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1781
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AssignmentListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def assignment(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.AssignmentContext)
            else:
                return self.getTypedRuleContext(sqlParser.AssignmentContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_assignmentList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAssignmentList'):
                return visitor.visitAssignmentList(self)
            else:
                return visitor.visitChildren(self)

    def assignmentList(self):
        localctx = sqlParser.AssignmentListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 106, self.RULE_assignmentList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1785
            self.assignment()
            self.state = 1790
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1786
                self.match(sqlParser.T__3)
                self.state = 1787
                self.assignment()
                self.state = 1792
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def EQ(self):
            return self.getToken(sqlParser.EQ, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_assignment

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAssignment'):
                return visitor.visitAssignment(self)
            else:
                return visitor.visitChildren(self)

    def assignment(self):
        localctx = sqlParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 108, self.RULE_assignment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1793
            localctx.key = self.multipartIdentifier()
            self.state = 1794
            self.match(sqlParser.EQ)
            self.state = 1795
            localctx.value = self.expression()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WhereClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def WHERE(self):
            return self.getToken(sqlParser.WHERE, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_whereClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWhereClause'):
                return visitor.visitWhereClause(self)
            else:
                return visitor.visitChildren(self)

    def whereClause(self):
        localctx = sqlParser.WhereClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 110, self.RULE_whereClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1797
            self.match(sqlParser.WHERE)
            self.state = 1798
            self.booleanExpression(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class HavingClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def HAVING(self):
            return self.getToken(sqlParser.HAVING, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_havingClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHavingClause'):
                return visitor.visitHavingClause(self)
            else:
                return visitor.visitChildren(self)

    def havingClause(self):
        localctx = sqlParser.HavingClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 112, self.RULE_havingClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1800
            self.match(sqlParser.HAVING)
            self.state = 1801
            self.booleanExpression(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class HintContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._hintStatement = None
            self.hintStatements = list()

        def hintStatement(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.HintStatementContext)
            else:
                return self.getTypedRuleContext(sqlParser.HintStatementContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_hint

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHint'):
                return visitor.visitHint(self)
            else:
                return visitor.visitChildren(self)

    def hint(self):
        localctx = sqlParser.HintContext(self, self._ctx, self.state)
        self.enterRule(localctx, 114, self.RULE_hint)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1803
            self.match(sqlParser.T__5)
            self.state = 1804
            localctx._hintStatement = self.hintStatement()
            localctx.hintStatements.append(localctx._hintStatement)
            self.state = 1811
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 217, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1806
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 216, self._ctx)
                    if la_ == 1:
                        self.state = 1805
                        self.match(sqlParser.T__3)
                    self.state = 1808
                    localctx._hintStatement = self.hintStatement()
                    localctx.hintStatements.append(localctx._hintStatement)
                self.state = 1813
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 217, self._ctx)
            self.state = 1814
            self.match(sqlParser.T__6)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class HintStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.hintName = None
            self._primaryExpression = None
            self.parameters = list()

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def primaryExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PrimaryExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.PrimaryExpressionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_hintStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHintStatement'):
                return visitor.visitHintStatement(self)
            else:
                return visitor.visitChildren(self)

    def hintStatement(self):
        localctx = sqlParser.HintStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 116, self.RULE_hintStatement)
        self._la = 0
        try:
            self.state = 1829
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 219, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1816
                localctx.hintName = self.identifier()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1817
                localctx.hintName = self.identifier()
                self.state = 1818
                self.match(sqlParser.T__1)
                self.state = 1819
                localctx._primaryExpression = self.primaryExpression(0)
                localctx.parameters.append(localctx._primaryExpression)
                self.state = 1824
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 1820
                    self.match(sqlParser.T__3)
                    self.state = 1821
                    localctx._primaryExpression = self.primaryExpression(0)
                    localctx.parameters.append(localctx._primaryExpression)
                    self.state = 1826
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1827
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FromClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def relation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.RelationContext)
            else:
                return self.getTypedRuleContext(sqlParser.RelationContext, i)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(sqlParser.LateralViewContext, i)

        def pivotClause(self):
            return self.getTypedRuleContext(sqlParser.PivotClauseContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_fromClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromClause'):
                return visitor.visitFromClause(self)
            else:
                return visitor.visitChildren(self)

    def fromClause(self):
        localctx = sqlParser.FromClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 118, self.RULE_fromClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1831
            self.match(sqlParser.FROM)
            self.state = 1832
            self.relation()
            self.state = 1837
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 220, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1833
                    self.match(sqlParser.T__3)
                    self.state = 1834
                    self.relation()
                self.state = 1839
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 220, self._ctx)
            self.state = 1843
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 221, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1840
                    self.lateralView()
                self.state = 1845
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 221, self._ctx)
            self.state = 1847
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 222, self._ctx)
            if la_ == 1:
                self.state = 1846
                self.pivotClause()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AggregationClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._expression = None
            self.groupingExpressions = list()
            self.kind = None

        def GROUP(self):
            return self.getToken(sqlParser.GROUP, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def SETS(self):
            return self.getToken(sqlParser.SETS, 0)

        def groupingSet(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.GroupingSetContext)
            else:
                return self.getTypedRuleContext(sqlParser.GroupingSetContext, i)

        def ROLLUP(self):
            return self.getToken(sqlParser.ROLLUP, 0)

        def CUBE(self):
            return self.getToken(sqlParser.CUBE, 0)

        def GROUPING(self):
            return self.getToken(sqlParser.GROUPING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_aggregationClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAggregationClause'):
                return visitor.visitAggregationClause(self)
            else:
                return visitor.visitChildren(self)

    def aggregationClause(self):
        localctx = sqlParser.AggregationClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 120, self.RULE_aggregationClause)
        self._la = 0
        try:
            self.state = 1893
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 227, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1849
                self.match(sqlParser.GROUP)
                self.state = 1850
                self.match(sqlParser.BY)
                self.state = 1851
                localctx._expression = self.expression()
                localctx.groupingExpressions.append(localctx._expression)
                self.state = 1856
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 223, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1852
                        self.match(sqlParser.T__3)
                        self.state = 1853
                        localctx._expression = self.expression()
                        localctx.groupingExpressions.append(localctx._expression)
                    self.state = 1858
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 223, self._ctx)
                self.state = 1876
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 225, self._ctx)
                if la_ == 1:
                    self.state = 1859
                    self.match(sqlParser.WITH)
                    self.state = 1860
                    localctx.kind = self.match(sqlParser.ROLLUP)
                elif la_ == 2:
                    self.state = 1861
                    self.match(sqlParser.WITH)
                    self.state = 1862
                    localctx.kind = self.match(sqlParser.CUBE)
                elif la_ == 3:
                    self.state = 1863
                    localctx.kind = self.match(sqlParser.GROUPING)
                    self.state = 1864
                    self.match(sqlParser.SETS)
                    self.state = 1865
                    self.match(sqlParser.T__1)
                    self.state = 1866
                    self.groupingSet()
                    self.state = 1871
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 1867
                        self.match(sqlParser.T__3)
                        self.state = 1868
                        self.groupingSet()
                        self.state = 1873
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 1874
                    self.match(sqlParser.T__2)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1878
                self.match(sqlParser.GROUP)
                self.state = 1879
                self.match(sqlParser.BY)
                self.state = 1880
                localctx.kind = self.match(sqlParser.GROUPING)
                self.state = 1881
                self.match(sqlParser.SETS)
                self.state = 1882
                self.match(sqlParser.T__1)
                self.state = 1883
                self.groupingSet()
                self.state = 1888
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 1884
                    self.match(sqlParser.T__3)
                    self.state = 1885
                    self.groupingSet()
                    self.state = 1890
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1891
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class GroupingSetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_groupingSet

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitGroupingSet'):
                return visitor.visitGroupingSet(self)
            else:
                return visitor.visitChildren(self)

    def groupingSet(self):
        localctx = sqlParser.GroupingSetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 122, self.RULE_groupingSet)
        self._la = 0
        try:
            self.state = 1908
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 230, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1895
                self.match(sqlParser.T__1)
                self.state = 1904
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 229, self._ctx)
                if la_ == 1:
                    self.state = 1896
                    self.expression()
                    self.state = 1901
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 1897
                        self.match(sqlParser.T__3)
                        self.state = 1898
                        self.expression()
                        self.state = 1903
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 1906
                self.match(sqlParser.T__2)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1907
                self.expression()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PivotClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.aggregates = None
            self._pivotValue = None
            self.pivotValues = list()

        def PIVOT(self):
            return self.getToken(sqlParser.PIVOT, 0)

        def FOR(self):
            return self.getToken(sqlParser.FOR, 0)

        def pivotColumn(self):
            return self.getTypedRuleContext(sqlParser.PivotColumnContext, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(sqlParser.NamedExpressionSeqContext, 0)

        def pivotValue(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.PivotValueContext)
            else:
                return self.getTypedRuleContext(sqlParser.PivotValueContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_pivotClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotClause'):
                return visitor.visitPivotClause(self)
            else:
                return visitor.visitChildren(self)

    def pivotClause(self):
        localctx = sqlParser.PivotClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 124, self.RULE_pivotClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1910
            self.match(sqlParser.PIVOT)
            self.state = 1911
            self.match(sqlParser.T__1)
            self.state = 1912
            localctx.aggregates = self.namedExpressionSeq()
            self.state = 1913
            self.match(sqlParser.FOR)
            self.state = 1914
            self.pivotColumn()
            self.state = 1915
            self.match(sqlParser.IN)
            self.state = 1916
            self.match(sqlParser.T__1)
            self.state = 1917
            localctx._pivotValue = self.pivotValue()
            localctx.pivotValues.append(localctx._pivotValue)
            self.state = 1922
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 1918
                self.match(sqlParser.T__3)
                self.state = 1919
                localctx._pivotValue = self.pivotValue()
                localctx.pivotValues.append(localctx._pivotValue)
                self.state = 1924
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1925
            self.match(sqlParser.T__2)
            self.state = 1926
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PivotColumnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._identifier = None
            self.identifiers = list()

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_pivotColumn

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotColumn'):
                return visitor.visitPivotColumn(self)
            else:
                return visitor.visitChildren(self)

    def pivotColumn(self):
        localctx = sqlParser.PivotColumnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 126, self.RULE_pivotColumn)
        self._la = 0
        try:
            self.state = 1940
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 233, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1928
                localctx._identifier = self.identifier()
                localctx.identifiers.append(localctx._identifier)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1929
                self.match(sqlParser.T__1)
                self.state = 1930
                localctx._identifier = self.identifier()
                localctx.identifiers.append(localctx._identifier)
                self.state = 1935
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 1931
                    self.match(sqlParser.T__3)
                    self.state = 1932
                    localctx._identifier = self.identifier()
                    localctx.identifiers.append(localctx._identifier)
                    self.state = 1937
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1938
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PivotValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_pivotValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotValue'):
                return visitor.visitPivotValue(self)
            else:
                return visitor.visitChildren(self)

    def pivotValue(self):
        localctx = sqlParser.PivotValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 128, self.RULE_pivotValue)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1942
            self.expression()
            self.state = 1947
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 235, self._ctx)
            if la_ == 1:
                self.state = 1944
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 234, self._ctx)
                if la_ == 1:
                    self.state = 1943
                    self.match(sqlParser.AS)
                self.state = 1946
                self.identifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class LateralViewContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.tblName = None
            self._identifier = None
            self.colName = list()

        def LATERAL(self):
            return self.getToken(sqlParser.LATERAL, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def OUTER(self):
            return self.getToken(sqlParser.OUTER, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_lateralView

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLateralView'):
                return visitor.visitLateralView(self)
            else:
                return visitor.visitChildren(self)

    def lateralView(self):
        localctx = sqlParser.LateralViewContext(self, self._ctx, self.state)
        self.enterRule(localctx, 130, self.RULE_lateralView)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1949
            self.match(sqlParser.LATERAL)
            self.state = 1950
            self.match(sqlParser.VIEW)
            self.state = 1952
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 236, self._ctx)
            if la_ == 1:
                self.state = 1951
                self.match(sqlParser.OUTER)
            self.state = 1954
            self.qualifiedName()
            self.state = 1955
            self.match(sqlParser.T__1)
            self.state = 1964
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 238, self._ctx)
            if la_ == 1:
                self.state = 1956
                self.expression()
                self.state = 1961
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 1957
                    self.match(sqlParser.T__3)
                    self.state = 1958
                    self.expression()
                    self.state = 1963
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
            self.state = 1966
            self.match(sqlParser.T__2)
            self.state = 1967
            localctx.tblName = self.identifier()
            self.state = 1979
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 241, self._ctx)
            if la_ == 1:
                self.state = 1969
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 239, self._ctx)
                if la_ == 1:
                    self.state = 1968
                    self.match(sqlParser.AS)
                self.state = 1971
                localctx._identifier = self.identifier()
                localctx.colName.append(localctx._identifier)
                self.state = 1976
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 240, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1972
                        self.match(sqlParser.T__3)
                        self.state = 1973
                        localctx._identifier = self.identifier()
                        localctx.colName.append(localctx._identifier)
                    self.state = 1978
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 240, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SetQuantifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DISTINCT(self):
            return self.getToken(sqlParser.DISTINCT, 0)

        def ALL(self):
            return self.getToken(sqlParser.ALL, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_setQuantifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetQuantifier'):
                return visitor.visitSetQuantifier(self)
            else:
                return visitor.visitChildren(self)

    def setQuantifier(self):
        localctx = sqlParser.SetQuantifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 132, self.RULE_setQuantifier)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1981
            _la = self._input.LA(1)
            if not (_la == 14 or _la == 74):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class RelationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def relationPrimary(self):
            return self.getTypedRuleContext(sqlParser.RelationPrimaryContext, 0)

        def joinRelation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.JoinRelationContext)
            else:
                return self.getTypedRuleContext(sqlParser.JoinRelationContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_relation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRelation'):
                return visitor.visitRelation(self)
            else:
                return visitor.visitChildren(self)

    def relation(self):
        localctx = sqlParser.RelationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 134, self.RULE_relation)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1983
            self.relationPrimary()
            self.state = 1987
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 242, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1984
                    self.joinRelation()
                self.state = 1989
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 242, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class JoinRelationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.right = None

        def JOIN(self):
            return self.getToken(sqlParser.JOIN, 0)

        def relationPrimary(self):
            return self.getTypedRuleContext(sqlParser.RelationPrimaryContext, 0)

        def joinType(self):
            return self.getTypedRuleContext(sqlParser.JoinTypeContext, 0)

        def joinCriteria(self):
            return self.getTypedRuleContext(sqlParser.JoinCriteriaContext, 0)

        def NATURAL(self):
            return self.getToken(sqlParser.NATURAL, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_joinRelation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinRelation'):
                return visitor.visitJoinRelation(self)
            else:
                return visitor.visitChildren(self)

    def joinRelation(self):
        localctx = sqlParser.JoinRelationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 136, self.RULE_joinRelation)
        try:
            self.state = 2001
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [18, 54, 101, 116, 125, 131, 196, 207]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1990
                self.joinType()
                self.state = 1991
                self.match(sqlParser.JOIN)
                self.state = 1992
                localctx.right = self.relationPrimary()
                self.state = 1994
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 243, self._ctx)
                if la_ == 1:
                    self.state = 1993
                    self.joinCriteria()
                pass
            elif token in [151]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1996
                self.match(sqlParser.NATURAL)
                self.state = 1997
                self.joinType()
                self.state = 1998
                self.match(sqlParser.JOIN)
                self.state = 1999
                localctx.right = self.relationPrimary()
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class JoinTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INNER(self):
            return self.getToken(sqlParser.INNER, 0)

        def CROSS(self):
            return self.getToken(sqlParser.CROSS, 0)

        def LEFT(self):
            return self.getToken(sqlParser.LEFT, 0)

        def OUTER(self):
            return self.getToken(sqlParser.OUTER, 0)

        def SEMI(self):
            return self.getToken(sqlParser.SEMI, 0)

        def RIGHT(self):
            return self.getToken(sqlParser.RIGHT, 0)

        def FULL(self):
            return self.getToken(sqlParser.FULL, 0)

        def ANTI(self):
            return self.getToken(sqlParser.ANTI, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_joinType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinType'):
                return visitor.visitJoinType(self)
            else:
                return visitor.visitChildren(self)

    def joinType(self):
        localctx = sqlParser.JoinTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 138, self.RULE_joinType)
        self._la = 0
        try:
            self.state = 2027
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 251, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2004
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 116:
                    self.state = 2003
                    self.match(sqlParser.INNER)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2006
                self.match(sqlParser.CROSS)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2007
                self.match(sqlParser.LEFT)
                self.state = 2009
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 164:
                    self.state = 2008
                    self.match(sqlParser.OUTER)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2012
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 131:
                    self.state = 2011
                    self.match(sqlParser.LEFT)
                self.state = 2014
                self.match(sqlParser.SEMI)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 2015
                self.match(sqlParser.RIGHT)
                self.state = 2017
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 164:
                    self.state = 2016
                    self.match(sqlParser.OUTER)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 2019
                self.match(sqlParser.FULL)
                self.state = 2021
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 164:
                    self.state = 2020
                    self.match(sqlParser.OUTER)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 2024
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 131:
                    self.state = 2023
                    self.match(sqlParser.LEFT)
                self.state = 2026
                self.match(sqlParser.ANTI)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class JoinCriteriaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_joinCriteria

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinCriteria'):
                return visitor.visitJoinCriteria(self)
            else:
                return visitor.visitChildren(self)

    def joinCriteria(self):
        localctx = sqlParser.JoinCriteriaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 140, self.RULE_joinCriteria)
        try:
            self.state = 2033
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [157]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2029
                self.match(sqlParser.ON)
                self.state = 2030
                self.booleanExpression(0)
                pass
            elif token in [255]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2031
                self.match(sqlParser.USING)
                self.state = 2032
                self.identifierList()
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SampleContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TABLESAMPLE(self):
            return self.getToken(sqlParser.TABLESAMPLE, 0)

        def sampleMethod(self):
            return self.getTypedRuleContext(sqlParser.SampleMethodContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_sample

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSample'):
                return visitor.visitSample(self)
            else:
                return visitor.visitChildren(self)

    def sample(self):
        localctx = sqlParser.SampleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 142, self.RULE_sample)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2035
            self.match(sqlParser.TABLESAMPLE)
            self.state = 2036
            self.match(sqlParser.T__1)
            self.state = 2038
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 253, self._ctx)
            if la_ == 1:
                self.state = 2037
                self.sampleMethod()
            self.state = 2040
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SampleMethodContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_sampleMethod

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class SampleByRowsContext(SampleMethodContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def ROWS(self):
            return self.getToken(sqlParser.ROWS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSampleByRows'):
                return visitor.visitSampleByRows(self)
            else:
                return visitor.visitChildren(self)

    class SampleByPercentileContext(SampleMethodContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.negativeSign = None
            self.percentage = None
            self.copyFrom(ctx)

        def PERCENTLIT(self):
            return self.getToken(sqlParser.PERCENTLIT, 0)

        def INTEGER_VALUE(self):
            return self.getToken(sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSampleByPercentile'):
                return visitor.visitSampleByPercentile(self)
            else:
                return visitor.visitChildren(self)

    class SampleByBucketContext(SampleMethodContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.sampleType = None
            self.numerator = None
            self.denominator = None
            self.copyFrom(ctx)

        def OUT(self):
            return self.getToken(sqlParser.OUT, 0)

        def OF(self):
            return self.getToken(sqlParser.OF, 0)

        def BUCKET(self):
            return self.getToken(sqlParser.BUCKET, 0)

        def INTEGER_VALUE(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.INTEGER_VALUE)
            else:
                return self.getToken(sqlParser.INTEGER_VALUE, i)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSampleByBucket'):
                return visitor.visitSampleByBucket(self)
            else:
                return visitor.visitChildren(self)

    class SampleByBytesContext(SampleMethodContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.bytes = None
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSampleByBytes'):
                return visitor.visitSampleByBytes(self)
            else:
                return visitor.visitChildren(self)

    def sampleMethod(self):
        localctx = sqlParser.SampleMethodContext(self, self._ctx, self.state)
        self.enterRule(localctx, 144, self.RULE_sampleMethod)
        self._la = 0
        try:
            self.state = 2066
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 257, self._ctx)
            if la_ == 1:
                localctx = sqlParser.SampleByPercentileContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2043
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2042
                    localctx.negativeSign = self.match(sqlParser.MINUS)
                self.state = 2045
                localctx.percentage = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 287 or _la == 289):
                    localctx.percentage = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2046
                self.match(sqlParser.PERCENTLIT)
                pass
            elif la_ == 2:
                localctx = sqlParser.SampleByRowsContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2047
                self.expression()
                self.state = 2048
                self.match(sqlParser.ROWS)
                pass
            elif la_ == 3:
                localctx = sqlParser.SampleByBucketContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2050
                localctx.sampleType = self.match(sqlParser.BUCKET)
                self.state = 2051
                localctx.numerator = self.match(sqlParser.INTEGER_VALUE)
                self.state = 2052
                self.match(sqlParser.OUT)
                self.state = 2053
                self.match(sqlParser.OF)
                self.state = 2054
                localctx.denominator = self.match(sqlParser.INTEGER_VALUE)
                self.state = 2063
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 157:
                    self.state = 2055
                    self.match(sqlParser.ON)
                    self.state = 2061
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 255, self._ctx)
                    if la_ == 1:
                        self.state = 2056
                        self.identifier()
                        pass
                    elif la_ == 2:
                        self.state = 2057
                        self.qualifiedName()
                        self.state = 2058
                        self.match(sqlParser.T__1)
                        self.state = 2059
                        self.match(sqlParser.T__2)
                        pass
                pass
            elif la_ == 4:
                localctx = sqlParser.SampleByBytesContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2065
                localctx.bytes = self.expression()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IdentifierListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifierSeq(self):
            return self.getTypedRuleContext(sqlParser.IdentifierSeqContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_identifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierList'):
                return visitor.visitIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def identifierList(self):
        localctx = sqlParser.IdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 146, self.RULE_identifierList)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2068
            self.match(sqlParser.T__1)
            self.state = 2069
            self.identifierSeq()
            self.state = 2070
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IdentifierSeqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._errorCapturingIdentifier = None
            self.ident = list()

        def errorCapturingIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_identifierSeq

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierSeq'):
                return visitor.visitIdentifierSeq(self)
            else:
                return visitor.visitChildren(self)

    def identifierSeq(self):
        localctx = sqlParser.IdentifierSeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 148, self.RULE_identifierSeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2072
            localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
            localctx.ident.append(localctx._errorCapturingIdentifier)
            self.state = 2077
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 258, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2073
                    self.match(sqlParser.T__3)
                    self.state = 2074
                    localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
                    localctx.ident.append(localctx._errorCapturingIdentifier)
                self.state = 2079
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 258, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OrderedIdentifierListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def orderedIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.OrderedIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.OrderedIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_orderedIdentifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOrderedIdentifierList'):
                return visitor.visitOrderedIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def orderedIdentifierList(self):
        localctx = sqlParser.OrderedIdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 150, self.RULE_orderedIdentifierList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2080
            self.match(sqlParser.T__1)
            self.state = 2081
            self.orderedIdentifier()
            self.state = 2086
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2082
                self.match(sqlParser.T__3)
                self.state = 2083
                self.orderedIdentifier()
                self.state = 2088
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2089
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OrderedIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.ident = None
            self.ordering = None

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def ASC(self):
            return self.getToken(sqlParser.ASC, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_orderedIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOrderedIdentifier'):
                return visitor.visitOrderedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def orderedIdentifier(self):
        localctx = sqlParser.OrderedIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 152, self.RULE_orderedIdentifier)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2091
            localctx.ident = self.errorCapturingIdentifier()
            self.state = 2093
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 23 or _la == 69:
                self.state = 2092
                localctx.ordering = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 23 or _la == 69):
                    localctx.ordering = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IdentifierCommentListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifierComment(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierCommentContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierCommentContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_identifierCommentList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierCommentList'):
                return visitor.visitIdentifierCommentList(self)
            else:
                return visitor.visitChildren(self)

    def identifierCommentList(self):
        localctx = sqlParser.IdentifierCommentListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 154, self.RULE_identifierCommentList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2095
            self.match(sqlParser.T__1)
            self.state = 2096
            self.identifierComment()
            self.state = 2101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2097
                self.match(sqlParser.T__3)
                self.state = 2098
                self.identifierComment()
                self.state = 2103
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2104
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IdentifierCommentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_identifierComment

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierComment'):
                return visitor.visitIdentifierComment(self)
            else:
                return visitor.visitChildren(self)

    def identifierComment(self):
        localctx = sqlParser.IdentifierCommentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 156, self.RULE_identifierComment)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2106
            self.identifier()
            self.state = 2108
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 45:
                self.state = 2107
                self.commentSpec()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class RelationPrimaryContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_relationPrimary

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class AliasedRelationContext(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def relation(self):
            return self.getTypedRuleContext(sqlParser.RelationContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def sample(self):
            return self.getTypedRuleContext(sqlParser.SampleContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAliasedRelation'):
                return visitor.visitAliasedRelation(self)
            else:
                return visitor.visitChildren(self)

    class AliasedQueryContext(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def sample(self):
            return self.getTypedRuleContext(sqlParser.SampleContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAliasedQuery'):
                return visitor.visitAliasedQuery(self)
            else:
                return visitor.visitChildren(self)

    class TableNameContext(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def sample(self):
            return self.getTypedRuleContext(sqlParser.SampleContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableName'):
                return visitor.visitTableName(self)
            else:
                return visitor.visitChildren(self)

    def relationPrimary(self):
        localctx = sqlParser.RelationPrimaryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 158, self.RULE_relationPrimary)
        try:
            self.state = 2132
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 266, self._ctx)
            if la_ == 1:
                localctx = sqlParser.TableNameContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2110
                self.multipartIdentifier()
                self.state = 2112
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 263, self._ctx)
                if la_ == 1:
                    self.state = 2111
                    self.sample()
                self.state = 2114
                self.tableAlias()
                pass
            elif la_ == 2:
                localctx = sqlParser.AliasedQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2116
                self.match(sqlParser.T__1)
                self.state = 2117
                self.query()
                self.state = 2118
                self.match(sqlParser.T__2)
                self.state = 2120
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 264, self._ctx)
                if la_ == 1:
                    self.state = 2119
                    self.sample()
                self.state = 2122
                self.tableAlias()
                pass
            elif la_ == 3:
                localctx = sqlParser.AliasedRelationContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2124
                self.match(sqlParser.T__1)
                self.state = 2125
                self.relation()
                self.state = 2126
                self.match(sqlParser.T__2)
                self.state = 2128
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 265, self._ctx)
                if la_ == 1:
                    self.state = 2127
                    self.sample()
                self.state = 2130
                self.tableAlias()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class InlineTableContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def VALUES(self):
            return self.getToken(sqlParser.VALUES, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_inlineTable

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInlineTable'):
                return visitor.visitInlineTable(self)
            else:
                return visitor.visitChildren(self)

    def inlineTable(self):
        localctx = sqlParser.InlineTableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 160, self.RULE_inlineTable)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2134
            self.match(sqlParser.VALUES)
            self.state = 2135
            self.expression()
            self.state = 2140
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 267, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2136
                    self.match(sqlParser.T__3)
                    self.state = 2137
                    self.expression()
                self.state = 2142
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 267, self._ctx)
            self.state = 2143
            self.tableAlias()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FunctionTableContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.funcName = None

        def tableAlias(self):
            return self.getTypedRuleContext(sqlParser.TableAliasContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_functionTable

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionTable'):
                return visitor.visitFunctionTable(self)
            else:
                return visitor.visitChildren(self)

    def functionTable(self):
        localctx = sqlParser.FunctionTableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 162, self.RULE_functionTable)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2145
            localctx.funcName = self.errorCapturingIdentifier()
            self.state = 2146
            self.match(sqlParser.T__1)
            self.state = 2155
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 269, self._ctx)
            if la_ == 1:
                self.state = 2147
                self.expression()
                self.state = 2152
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 2148
                    self.match(sqlParser.T__3)
                    self.state = 2149
                    self.expression()
                    self.state = 2154
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
            self.state = 2157
            self.match(sqlParser.T__2)
            self.state = 2158
            self.tableAlias()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TableAliasContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def strictIdentifier(self):
            return self.getTypedRuleContext(sqlParser.StrictIdentifierContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_tableAlias

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableAlias'):
                return visitor.visitTableAlias(self)
            else:
                return visitor.visitChildren(self)

    def tableAlias(self):
        localctx = sqlParser.TableAliasContext(self, self._ctx, self.state)
        self.enterRule(localctx, 164, self.RULE_tableAlias)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2167
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 272, self._ctx)
            if la_ == 1:
                self.state = 2161
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 270, self._ctx)
                if la_ == 1:
                    self.state = 2160
                    self.match(sqlParser.AS)
                self.state = 2163
                self.strictIdentifier()
                self.state = 2165
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 271, self._ctx)
                if la_ == 1:
                    self.state = 2164
                    self.identifierList()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class RowFormatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_rowFormat

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class RowFormatSerdeContext(RowFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.name = None
            self.props = None
            self.copyFrom(ctx)

        def ROW(self):
            return self.getToken(sqlParser.ROW, 0)

        def FORMAT(self):
            return self.getToken(sqlParser.FORMAT, 0)

        def SERDE(self):
            return self.getToken(sqlParser.SERDE, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(sqlParser.TablePropertyListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRowFormatSerde'):
                return visitor.visitRowFormatSerde(self)
            else:
                return visitor.visitChildren(self)

    class RowFormatDelimitedContext(RowFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.fieldsTerminatedBy = None
            self.escapedBy = None
            self.collectionItemsTerminatedBy = None
            self.keysTerminatedBy = None
            self.linesSeparatedBy = None
            self.nullDefinedAs = None
            self.copyFrom(ctx)

        def ROW(self):
            return self.getToken(sqlParser.ROW, 0)

        def FORMAT(self):
            return self.getToken(sqlParser.FORMAT, 0)

        def DELIMITED(self):
            return self.getToken(sqlParser.DELIMITED, 0)

        def FIELDS(self):
            return self.getToken(sqlParser.FIELDS, 0)

        def TERMINATED(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.TERMINATED)
            else:
                return self.getToken(sqlParser.TERMINATED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def COLLECTION(self):
            return self.getToken(sqlParser.COLLECTION, 0)

        def ITEMS(self):
            return self.getToken(sqlParser.ITEMS, 0)

        def MAP(self):
            return self.getToken(sqlParser.MAP, 0)

        def KEYS(self):
            return self.getToken(sqlParser.KEYS, 0)

        def LINES(self):
            return self.getToken(sqlParser.LINES, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def DEFINED(self):
            return self.getToken(sqlParser.DEFINED, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.STRING)
            else:
                return self.getToken(sqlParser.STRING, i)

        def ESCAPED(self):
            return self.getToken(sqlParser.ESCAPED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRowFormatDelimited'):
                return visitor.visitRowFormatDelimited(self)
            else:
                return visitor.visitChildren(self)

    def rowFormat(self):
        localctx = sqlParser.RowFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 166, self.RULE_rowFormat)
        try:
            self.state = 2218
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 280, self._ctx)
            if la_ == 1:
                localctx = sqlParser.RowFormatSerdeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2169
                self.match(sqlParser.ROW)
                self.state = 2170
                self.match(sqlParser.FORMAT)
                self.state = 2171
                self.match(sqlParser.SERDE)
                self.state = 2172
                localctx.name = self.match(sqlParser.STRING)
                self.state = 2176
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 273, self._ctx)
                if la_ == 1:
                    self.state = 2173
                    self.match(sqlParser.WITH)
                    self.state = 2174
                    self.match(sqlParser.SERDEPROPERTIES)
                    self.state = 2175
                    localctx.props = self.tablePropertyList()
                pass
            elif la_ == 2:
                localctx = sqlParser.RowFormatDelimitedContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2178
                self.match(sqlParser.ROW)
                self.state = 2179
                self.match(sqlParser.FORMAT)
                self.state = 2180
                self.match(sqlParser.DELIMITED)
                self.state = 2190
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 275, self._ctx)
                if la_ == 1:
                    self.state = 2181
                    self.match(sqlParser.FIELDS)
                    self.state = 2182
                    self.match(sqlParser.TERMINATED)
                    self.state = 2183
                    self.match(sqlParser.BY)
                    self.state = 2184
                    localctx.fieldsTerminatedBy = self.match(sqlParser.STRING)
                    self.state = 2188
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 274, self._ctx)
                    if la_ == 1:
                        self.state = 2185
                        self.match(sqlParser.ESCAPED)
                        self.state = 2186
                        self.match(sqlParser.BY)
                        self.state = 2187
                        localctx.escapedBy = self.match(sqlParser.STRING)
                self.state = 2197
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 276, self._ctx)
                if la_ == 1:
                    self.state = 2192
                    self.match(sqlParser.COLLECTION)
                    self.state = 2193
                    self.match(sqlParser.ITEMS)
                    self.state = 2194
                    self.match(sqlParser.TERMINATED)
                    self.state = 2195
                    self.match(sqlParser.BY)
                    self.state = 2196
                    localctx.collectionItemsTerminatedBy = self.match(sqlParser.STRING)
                self.state = 2204
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 277, self._ctx)
                if la_ == 1:
                    self.state = 2199
                    self.match(sqlParser.MAP)
                    self.state = 2200
                    self.match(sqlParser.KEYS)
                    self.state = 2201
                    self.match(sqlParser.TERMINATED)
                    self.state = 2202
                    self.match(sqlParser.BY)
                    self.state = 2203
                    localctx.keysTerminatedBy = self.match(sqlParser.STRING)
                self.state = 2210
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 278, self._ctx)
                if la_ == 1:
                    self.state = 2206
                    self.match(sqlParser.LINES)
                    self.state = 2207
                    self.match(sqlParser.TERMINATED)
                    self.state = 2208
                    self.match(sqlParser.BY)
                    self.state = 2209
                    localctx.linesSeparatedBy = self.match(sqlParser.STRING)
                self.state = 2216
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 279, self._ctx)
                if la_ == 1:
                    self.state = 2212
                    self.match(sqlParser.NULL)
                    self.state = 2213
                    self.match(sqlParser.DEFINED)
                    self.state = 2214
                    self.match(sqlParser.AS)
                    self.state = 2215
                    localctx.nullDefinedAs = self.match(sqlParser.STRING)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MultipartIdentifierListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_multipartIdentifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultipartIdentifierList'):
                return visitor.visitMultipartIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def multipartIdentifierList(self):
        localctx = sqlParser.MultipartIdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 168, self.RULE_multipartIdentifierList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2220
            self.multipartIdentifier()
            self.state = 2225
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2221
                self.match(sqlParser.T__3)
                self.state = 2222
                self.multipartIdentifier()
                self.state = 2227
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MultipartIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._errorCapturingIdentifier = None
            self.parts = list()

        def errorCapturingIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_multipartIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultipartIdentifier'):
                return visitor.visitMultipartIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def multipartIdentifier(self):
        localctx = sqlParser.MultipartIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 170, self.RULE_multipartIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2228
            localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
            localctx.parts.append(localctx._errorCapturingIdentifier)
            self.state = 2233
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 282, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2229
                    self.match(sqlParser.T__4)
                    self.state = 2230
                    localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
                    localctx.parts.append(localctx._errorCapturingIdentifier)
                self.state = 2235
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 282, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TableIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.db = None
            self.table = None

        def errorCapturingIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_tableIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableIdentifier'):
                return visitor.visitTableIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def tableIdentifier(self):
        localctx = sqlParser.TableIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 172, self.RULE_tableIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2239
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 283, self._ctx)
            if la_ == 1:
                self.state = 2236
                localctx.db = self.errorCapturingIdentifier()
                self.state = 2237
                self.match(sqlParser.T__4)
            self.state = 2241
            localctx.table = self.errorCapturingIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FunctionIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.db = None
            self.function = None

        def errorCapturingIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_functionIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionIdentifier'):
                return visitor.visitFunctionIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def functionIdentifier(self):
        localctx = sqlParser.FunctionIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 174, self.RULE_functionIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2246
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 284, self._ctx)
            if la_ == 1:
                self.state = 2243
                localctx.db = self.errorCapturingIdentifier()
                self.state = 2244
                self.match(sqlParser.T__4)
            self.state = 2248
            localctx.function = self.errorCapturingIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NamedExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def identifierList(self):
            return self.getTypedRuleContext(sqlParser.IdentifierListContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_namedExpression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedExpression'):
                return visitor.visitNamedExpression(self)
            else:
                return visitor.visitChildren(self)

    def namedExpression(self):
        localctx = sqlParser.NamedExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 176, self.RULE_namedExpression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2250
            self.expression()
            self.state = 2258
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 287, self._ctx)
            if la_ == 1:
                self.state = 2252
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 285, self._ctx)
                if la_ == 1:
                    self.state = 2251
                    self.match(sqlParser.AS)
                self.state = 2256
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 286, self._ctx)
                if la_ == 1:
                    self.state = 2254
                    localctx.name = self.errorCapturingIdentifier()
                    pass
                elif la_ == 2:
                    self.state = 2255
                    self.identifierList()
                    pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NamedExpressionSeqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def namedExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.NamedExpressionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_namedExpressionSeq

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedExpressionSeq'):
                return visitor.visitNamedExpressionSeq(self)
            else:
                return visitor.visitChildren(self)

    def namedExpressionSeq(self):
        localctx = sqlParser.NamedExpressionSeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 178, self.RULE_namedExpressionSeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2260
            self.namedExpression()
            self.state = 2265
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 288, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2261
                    self.match(sqlParser.T__3)
                    self.state = 2262
                    self.namedExpression()
                self.state = 2267
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 288, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransformListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._transform = None
            self.transforms = list()

        def transform(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TransformContext)
            else:
                return self.getTypedRuleContext(sqlParser.TransformContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_transformList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformList'):
                return visitor.visitTransformList(self)
            else:
                return visitor.visitChildren(self)

    def transformList(self):
        localctx = sqlParser.TransformListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 180, self.RULE_transformList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2268
            self.match(sqlParser.T__1)
            self.state = 2269
            localctx._transform = self.transform()
            localctx.transforms.append(localctx._transform)
            self.state = 2274
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2270
                self.match(sqlParser.T__3)
                self.state = 2271
                localctx._transform = self.transform()
                localctx.transforms.append(localctx._transform)
                self.state = 2276
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2277
            self.match(sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransformContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_transform

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class IdentityTransformContext(TransformContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentityTransform'):
                return visitor.visitIdentityTransform(self)
            else:
                return visitor.visitChildren(self)

    class ApplyTransformContext(TransformContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.transformName = None
            self._transformArgument = None
            self.argument = list()
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def transformArgument(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.TransformArgumentContext)
            else:
                return self.getTypedRuleContext(sqlParser.TransformArgumentContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitApplyTransform'):
                return visitor.visitApplyTransform(self)
            else:
                return visitor.visitChildren(self)

    def transform(self):
        localctx = sqlParser.TransformContext(self, self._ctx, self.state)
        self.enterRule(localctx, 182, self.RULE_transform)
        self._la = 0
        try:
            self.state = 2292
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 291, self._ctx)
            if la_ == 1:
                localctx = sqlParser.IdentityTransformContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2279
                self.qualifiedName()
                pass
            elif la_ == 2:
                localctx = sqlParser.ApplyTransformContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2280
                localctx.transformName = self.identifier()
                self.state = 2281
                self.match(sqlParser.T__1)
                self.state = 2282
                localctx._transformArgument = self.transformArgument()
                localctx.argument.append(localctx._transformArgument)
                self.state = 2287
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 2283
                    self.match(sqlParser.T__3)
                    self.state = 2284
                    localctx._transformArgument = self.transformArgument()
                    localctx.argument.append(localctx._transformArgument)
                    self.state = 2289
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2290
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransformArgumentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def constant(self):
            return self.getTypedRuleContext(sqlParser.ConstantContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_transformArgument

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformArgument'):
                return visitor.visitTransformArgument(self)
            else:
                return visitor.visitChildren(self)

    def transformArgument(self):
        localctx = sqlParser.TransformArgumentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 184, self.RULE_transformArgument)
        try:
            self.state = 2296
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 292, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2294
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2295
                self.constant()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_expression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExpression'):
                return visitor.visitExpression(self)
            else:
                return visitor.visitChildren(self)

    def expression(self):
        localctx = sqlParser.ExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 186, self.RULE_expression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2298
            self.booleanExpression(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class BooleanExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_booleanExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class LogicalNotContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLogicalNot'):
                return visitor.visitLogicalNot(self)
            else:
                return visitor.visitChildren(self)

    class PredicatedContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def valueExpression(self):
            return self.getTypedRuleContext(sqlParser.ValueExpressionContext, 0)

        def predicate(self):
            return self.getTypedRuleContext(sqlParser.PredicateContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPredicated'):
                return visitor.visitPredicated(self)
            else:
                return visitor.visitChildren(self)

    class ExistsContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExists'):
                return visitor.visitExists(self)
            else:
                return visitor.visitChildren(self)

    class LogicalBinaryContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.operator = None
            self.right = None
            self.copyFrom(ctx)

        def booleanExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.BooleanExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, i)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLogicalBinary'):
                return visitor.visitLogicalBinary(self)
            else:
                return visitor.visitChildren(self)

    def booleanExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = sqlParser.BooleanExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 188
        self.enterRecursionRule(localctx, 188, self.RULE_booleanExpression, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2312
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 294, self._ctx)
            if la_ == 1:
                localctx = sqlParser.LogicalNotContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2301
                self.match(sqlParser.NOT)
                self.state = 2302
                self.booleanExpression(5)
                pass
            elif la_ == 2:
                localctx = sqlParser.ExistsContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2303
                self.match(sqlParser.EXISTS)
                self.state = 2304
                self.match(sqlParser.T__1)
                self.state = 2305
                self.query()
                self.state = 2306
                self.match(sqlParser.T__2)
                pass
            elif la_ == 3:
                localctx = sqlParser.PredicatedContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2308
                self.valueExpression(0)
                self.state = 2310
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 293, self._ctx)
                if la_ == 1:
                    self.state = 2309
                    self.predicate()
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 2322
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 296, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 2320
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 295, self._ctx)
                    if la_ == 1:
                        localctx = sqlParser.LogicalBinaryContext(self, sqlParser.BooleanExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_booleanExpression)
                        self.state = 2314
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 2315
                        localctx.operator = self.match(sqlParser.AND)
                        self.state = 2316
                        localctx.right = self.booleanExpression(3)
                        pass
                    elif la_ == 2:
                        localctx = sqlParser.LogicalBinaryContext(self, sqlParser.BooleanExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_booleanExpression)
                        self.state = 2317
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 2318
                        localctx.operator = self.match(sqlParser.OR)
                        self.state = 2319
                        localctx.right = self.booleanExpression(2)
                        pass
                self.state = 2324
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 296, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class PredicateContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.kind = None
            self.lower = None
            self.upper = None
            self.pattern = None
            self.quantifier = None
            self.escapeChar = None
            self.right = None

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def BETWEEN(self):
            return self.getToken(sqlParser.BETWEEN, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def RLIKE(self):
            return self.getToken(sqlParser.RLIKE, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def ANY(self):
            return self.getToken(sqlParser.ANY, 0)

        def SOME(self):
            return self.getToken(sqlParser.SOME, 0)

        def ALL(self):
            return self.getToken(sqlParser.ALL, 0)

        def ESCAPE(self):
            return self.getToken(sqlParser.ESCAPE, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def IS(self):
            return self.getToken(sqlParser.IS, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def TRUE(self):
            return self.getToken(sqlParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(sqlParser.FALSE, 0)

        def UNKNOWN(self):
            return self.getToken(sqlParser.UNKNOWN, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def DISTINCT(self):
            return self.getToken(sqlParser.DISTINCT, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_predicate

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPredicate'):
                return visitor.visitPredicate(self)
            else:
                return visitor.visitChildren(self)

    def predicate(self):
        localctx = sqlParser.PredicateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 190, self.RULE_predicate)
        self._la = 0
        try:
            self.state = 2407
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 310, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2326
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2325
                    self.match(sqlParser.NOT)
                self.state = 2328
                localctx.kind = self.match(sqlParser.BETWEEN)
                self.state = 2329
                localctx.lower = self.valueExpression(0)
                self.state = 2330
                self.match(sqlParser.AND)
                self.state = 2331
                localctx.upper = self.valueExpression(0)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2334
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2333
                    self.match(sqlParser.NOT)
                self.state = 2336
                localctx.kind = self.match(sqlParser.IN)
                self.state = 2337
                self.match(sqlParser.T__1)
                self.state = 2338
                self.expression()
                self.state = 2343
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 4:
                    self.state = 2339
                    self.match(sqlParser.T__3)
                    self.state = 2340
                    self.expression()
                    self.state = 2345
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2346
                self.match(sqlParser.T__2)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2349
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2348
                    self.match(sqlParser.NOT)
                self.state = 2351
                localctx.kind = self.match(sqlParser.IN)
                self.state = 2352
                self.match(sqlParser.T__1)
                self.state = 2353
                self.query()
                self.state = 2354
                self.match(sqlParser.T__2)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2357
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2356
                    self.match(sqlParser.NOT)
                self.state = 2359
                localctx.kind = self.match(sqlParser.RLIKE)
                self.state = 2360
                localctx.pattern = self.valueExpression(0)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 2362
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2361
                    self.match(sqlParser.NOT)
                self.state = 2364
                localctx.kind = self.match(sqlParser.LIKE)
                self.state = 2365
                localctx.quantifier = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 14 or _la == 19 or _la == 217):
                    localctx.quantifier = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2379
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 304, self._ctx)
                if la_ == 1:
                    self.state = 2366
                    self.match(sqlParser.T__1)
                    self.state = 2367
                    self.match(sqlParser.T__2)
                    pass
                elif la_ == 2:
                    self.state = 2368
                    self.match(sqlParser.T__1)
                    self.state = 2369
                    self.expression()
                    self.state = 2374
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 2370
                        self.match(sqlParser.T__3)
                        self.state = 2371
                        self.expression()
                        self.state = 2376
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 2377
                    self.match(sqlParser.T__2)
                    pass
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 2382
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2381
                    self.match(sqlParser.NOT)
                self.state = 2384
                localctx.kind = self.match(sqlParser.LIKE)
                self.state = 2385
                localctx.pattern = self.valueExpression(0)
                self.state = 2388
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 306, self._ctx)
                if la_ == 1:
                    self.state = 2386
                    self.match(sqlParser.ESCAPE)
                    self.state = 2387
                    localctx.escapeChar = self.match(sqlParser.STRING)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 2390
                self.match(sqlParser.IS)
                self.state = 2392
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2391
                    self.match(sqlParser.NOT)
                self.state = 2394
                localctx.kind = self.match(sqlParser.NULL)
                pass
            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 2395
                self.match(sqlParser.IS)
                self.state = 2397
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2396
                    self.match(sqlParser.NOT)
                self.state = 2399
                localctx.kind = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 89 or _la == 241 or _la == 249):
                    localctx.kind = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 2400
                self.match(sqlParser.IS)
                self.state = 2402
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 153:
                    self.state = 2401
                    self.match(sqlParser.NOT)
                self.state = 2404
                localctx.kind = self.match(sqlParser.DISTINCT)
                self.state = 2405
                self.match(sqlParser.FROM)
                self.state = 2406
                localctx.right = self.valueExpression(0)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ValueExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_valueExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ValueExpressionDefaultContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def primaryExpression(self):
            return self.getTypedRuleContext(sqlParser.PrimaryExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitValueExpressionDefault'):
                return visitor.visitValueExpressionDefault(self)
            else:
                return visitor.visitChildren(self)

    class ComparisonContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.right = None
            self.copyFrom(ctx)

        def comparisonOperator(self):
            return self.getTypedRuleContext(sqlParser.ComparisonOperatorContext, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComparison'):
                return visitor.visitComparison(self)
            else:
                return visitor.visitChildren(self)

    class ArithmeticBinaryContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.operator = None
            self.right = None
            self.copyFrom(ctx)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def ASTERISK(self):
            return self.getToken(sqlParser.ASTERISK, 0)

        def SLASH(self):
            return self.getToken(sqlParser.SLASH, 0)

        def PERCENT(self):
            return self.getToken(sqlParser.PERCENT, 0)

        def DIV(self):
            return self.getToken(sqlParser.DIV, 0)

        def PLUS(self):
            return self.getToken(sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def CONCAT_PIPE(self):
            return self.getToken(sqlParser.CONCAT_PIPE, 0)

        def AMPERSAND(self):
            return self.getToken(sqlParser.AMPERSAND, 0)

        def HAT(self):
            return self.getToken(sqlParser.HAT, 0)

        def PIPE(self):
            return self.getToken(sqlParser.PIPE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticBinary'):
                return visitor.visitArithmeticBinary(self)
            else:
                return visitor.visitChildren(self)

    class ArithmeticUnaryContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.operator = None
            self.copyFrom(ctx)

        def valueExpression(self):
            return self.getTypedRuleContext(sqlParser.ValueExpressionContext, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def PLUS(self):
            return self.getToken(sqlParser.PLUS, 0)

        def TILDE(self):
            return self.getToken(sqlParser.TILDE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticUnary'):
                return visitor.visitArithmeticUnary(self)
            else:
                return visitor.visitChildren(self)

    def valueExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = sqlParser.ValueExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 192
        self.enterRecursionRule(localctx, 192, self.RULE_valueExpression, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2413
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 311, self._ctx)
            if la_ == 1:
                localctx = sqlParser.ValueExpressionDefaultContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2410
                self.primaryExpression(0)
                pass
            elif la_ == 2:
                localctx = sqlParser.ArithmeticUnaryContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2411
                localctx.operator = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la - 272 & ~63 == 0 and 1 << _la - 272 & 67 != 0):
                    localctx.operator = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2412
                self.valueExpression(7)
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 2436
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 313, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 2434
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 312, self._ctx)
                    if la_ == 1:
                        localctx = sqlParser.ArithmeticBinaryContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2415
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 6)')
                        self.state = 2416
                        localctx.operator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la - 274 & ~63 == 0 and 1 << _la - 274 & 15 != 0):
                            localctx.operator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2417
                        localctx.right = self.valueExpression(7)
                        pass
                    elif la_ == 2:
                        localctx = sqlParser.ArithmeticBinaryContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2418
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 5)')
                        self.state = 2419
                        localctx.operator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la - 272 & ~63 == 0 and 1 << _la - 272 & 515 != 0):
                            localctx.operator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2420
                        localctx.right = self.valueExpression(6)
                        pass
                    elif la_ == 3:
                        localctx = sqlParser.ArithmeticBinaryContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2421
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 4)')
                        self.state = 2422
                        localctx.operator = self.match(sqlParser.AMPERSAND)
                        self.state = 2423
                        localctx.right = self.valueExpression(5)
                        pass
                    elif la_ == 4:
                        localctx = sqlParser.ArithmeticBinaryContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2424
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 3)')
                        self.state = 2425
                        localctx.operator = self.match(sqlParser.HAT)
                        self.state = 2426
                        localctx.right = self.valueExpression(4)
                        pass
                    elif la_ == 5:
                        localctx = sqlParser.ArithmeticBinaryContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2427
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 2428
                        localctx.operator = self.match(sqlParser.PIPE)
                        self.state = 2429
                        localctx.right = self.valueExpression(3)
                        pass
                    elif la_ == 6:
                        localctx = sqlParser.ComparisonContext(self, sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 2430
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 2431
                        self.comparisonOperator()
                        self.state = 2432
                        localctx.right = self.valueExpression(2)
                        pass
                self.state = 2438
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 313, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class PrimaryExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_primaryExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class StructContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self._namedExpression = None
            self.argument = list()
            self.copyFrom(ctx)

        def STRUCT(self):
            return self.getToken(sqlParser.STRUCT, 0)

        def namedExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.NamedExpressionContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStruct'):
                return visitor.visitStruct(self)
            else:
                return visitor.visitChildren(self)

    class DereferenceContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.base = None
            self.fieldName = None
            self.copyFrom(ctx)

        def primaryExpression(self):
            return self.getTypedRuleContext(sqlParser.PrimaryExpressionContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDereference'):
                return visitor.visitDereference(self)
            else:
                return visitor.visitChildren(self)

    class SimpleCaseContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.value = None
            self.elseExpression = None
            self.copyFrom(ctx)

        def CASE(self):
            return self.getToken(sqlParser.CASE, 0)

        def END(self):
            return self.getToken(sqlParser.END, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def whenClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.WhenClauseContext)
            else:
                return self.getTypedRuleContext(sqlParser.WhenClauseContext, i)

        def ELSE(self):
            return self.getToken(sqlParser.ELSE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSimpleCase'):
                return visitor.visitSimpleCase(self)
            else:
                return visitor.visitChildren(self)

    class ColumnReferenceContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColumnReference'):
                return visitor.visitColumnReference(self)
            else:
                return visitor.visitChildren(self)

    class RowConstructorContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def namedExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.NamedExpressionContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRowConstructor'):
                return visitor.visitRowConstructor(self)
            else:
                return visitor.visitChildren(self)

    class LastContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def LAST(self):
            return self.getToken(sqlParser.LAST, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def IGNORE(self):
            return self.getToken(sqlParser.IGNORE, 0)

        def NULLS(self):
            return self.getToken(sqlParser.NULLS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLast'):
                return visitor.visitLast(self)
            else:
                return visitor.visitChildren(self)

    class StarContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def ASTERISK(self):
            return self.getToken(sqlParser.ASTERISK, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStar'):
                return visitor.visitStar(self)
            else:
                return visitor.visitChildren(self)

    class OverlayContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.qpdinput = None
            self.replace = None
            self.position = None
            self.length = None
            self.copyFrom(ctx)

        def OVERLAY(self):
            return self.getToken(sqlParser.OVERLAY, 0)

        def PLACING(self):
            return self.getToken(sqlParser.PLACING, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def FOR(self):
            return self.getToken(sqlParser.FOR, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOverlay'):
                return visitor.visitOverlay(self)
            else:
                return visitor.visitChildren(self)

    class SubscriptContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.value = None
            self.index = None
            self.copyFrom(ctx)

        def primaryExpression(self):
            return self.getTypedRuleContext(sqlParser.PrimaryExpressionContext, 0)

        def valueExpression(self):
            return self.getTypedRuleContext(sqlParser.ValueExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSubscript'):
                return visitor.visitSubscript(self)
            else:
                return visitor.visitChildren(self)

    class SubqueryExpressionContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def query(self):
            return self.getTypedRuleContext(sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSubqueryExpression'):
                return visitor.visitSubqueryExpression(self)
            else:
                return visitor.visitChildren(self)

    class SubstringContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.qpdstr = None
            self.pos = None
            self.qpdlen = None
            self.copyFrom(ctx)

        def SUBSTR(self):
            return self.getToken(sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(sqlParser.SUBSTRING, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def FOR(self):
            return self.getToken(sqlParser.FOR, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSubstring'):
                return visitor.visitSubstring(self)
            else:
                return visitor.visitChildren(self)

    class CurrentDatetimeContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.name = None
            self.copyFrom(ctx)

        def CURRENT_DATE(self):
            return self.getToken(sqlParser.CURRENT_DATE, 0)

        def CURRENT_TIMESTAMP(self):
            return self.getToken(sqlParser.CURRENT_TIMESTAMP, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCurrentDatetime'):
                return visitor.visitCurrentDatetime(self)
            else:
                return visitor.visitChildren(self)

    class CastContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def CAST(self):
            return self.getToken(sqlParser.CAST, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCast'):
                return visitor.visitCast(self)
            else:
                return visitor.visitChildren(self)

    class ConstantDefaultContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def constant(self):
            return self.getTypedRuleContext(sqlParser.ConstantContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitConstantDefault'):
                return visitor.visitConstantDefault(self)
            else:
                return visitor.visitChildren(self)

    class LambdaContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLambda'):
                return visitor.visitLambda(self)
            else:
                return visitor.visitChildren(self)

    class ParenthesizedExpressionContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitParenthesizedExpression'):
                return visitor.visitParenthesizedExpression(self)
            else:
                return visitor.visitChildren(self)

    class ExtractContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.field = None
            self.source = None
            self.copyFrom(ctx)

        def EXTRACT(self):
            return self.getToken(sqlParser.EXTRACT, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def valueExpression(self):
            return self.getTypedRuleContext(sqlParser.ValueExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExtract'):
                return visitor.visitExtract(self)
            else:
                return visitor.visitChildren(self)

    class TrimContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.trimOption = None
            self.trimStr = None
            self.srcStr = None
            self.copyFrom(ctx)

        def TRIM(self):
            return self.getToken(sqlParser.TRIM, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def BOTH(self):
            return self.getToken(sqlParser.BOTH, 0)

        def LEADING(self):
            return self.getToken(sqlParser.LEADING, 0)

        def TRAILING(self):
            return self.getToken(sqlParser.TRAILING, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTrim'):
                return visitor.visitTrim(self)
            else:
                return visitor.visitChildren(self)

    class FunctionCallContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self._expression = None
            self.argument = list()
            self.where = None
            self.copyFrom(ctx)

        def functionName(self):
            return self.getTypedRuleContext(sqlParser.FunctionNameContext, 0)

        def FILTER(self):
            return self.getToken(sqlParser.FILTER, 0)

        def WHERE(self):
            return self.getToken(sqlParser.WHERE, 0)

        def OVER(self):
            return self.getToken(sqlParser.OVER, 0)

        def windowSpec(self):
            return self.getTypedRuleContext(sqlParser.WindowSpecContext, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def booleanExpression(self):
            return self.getTypedRuleContext(sqlParser.BooleanExpressionContext, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(sqlParser.SetQuantifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionCall'):
                return visitor.visitFunctionCall(self)
            else:
                return visitor.visitChildren(self)

    class SearchedCaseContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.elseExpression = None
            self.copyFrom(ctx)

        def CASE(self):
            return self.getToken(sqlParser.CASE, 0)

        def END(self):
            return self.getToken(sqlParser.END, 0)

        def whenClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.WhenClauseContext)
            else:
                return self.getTypedRuleContext(sqlParser.WhenClauseContext, i)

        def ELSE(self):
            return self.getToken(sqlParser.ELSE, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSearchedCase'):
                return visitor.visitSearchedCase(self)
            else:
                return visitor.visitChildren(self)

    class PositionContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.substr = None
            self.qpdstr = None
            self.copyFrom(ctx)

        def POSITION(self):
            return self.getToken(sqlParser.POSITION, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ValueExpressionContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPosition'):
                return visitor.visitPosition(self)
            else:
                return visitor.visitChildren(self)

    class FirstContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def FIRST(self):
            return self.getToken(sqlParser.FIRST, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def IGNORE(self):
            return self.getToken(sqlParser.IGNORE, 0)

        def NULLS(self):
            return self.getToken(sqlParser.NULLS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFirst'):
                return visitor.visitFirst(self)
            else:
                return visitor.visitChildren(self)

    def primaryExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = sqlParser.PrimaryExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 194
        self.enterRecursionRule(localctx, 194, self.RULE_primaryExpression, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2623
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 333, self._ctx)
            if la_ == 1:
                localctx = sqlParser.CurrentDatetimeContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2440
                localctx.name = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 57 or _la == 59):
                    localctx.name = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                localctx = sqlParser.SearchedCaseContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2441
                self.match(sqlParser.CASE)
                self.state = 2443
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 2442
                    self.whenClause()
                    self.state = 2445
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 259:
                        break
                self.state = 2449
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 77:
                    self.state = 2447
                    self.match(sqlParser.ELSE)
                    self.state = 2448
                    localctx.elseExpression = self.expression()
                self.state = 2451
                self.match(sqlParser.END)
                pass
            elif la_ == 3:
                localctx = sqlParser.SimpleCaseContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2453
                self.match(sqlParser.CASE)
                self.state = 2454
                localctx.value = self.expression()
                self.state = 2456
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 2455
                    self.whenClause()
                    self.state = 2458
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 259:
                        break
                self.state = 2462
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 77:
                    self.state = 2460
                    self.match(sqlParser.ELSE)
                    self.state = 2461
                    localctx.elseExpression = self.expression()
                self.state = 2464
                self.match(sqlParser.END)
                pass
            elif la_ == 4:
                localctx = sqlParser.CastContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2466
                self.match(sqlParser.CAST)
                self.state = 2467
                self.match(sqlParser.T__1)
                self.state = 2468
                self.expression()
                self.state = 2469
                self.match(sqlParser.AS)
                self.state = 2470
                self.dataType()
                self.state = 2471
                self.match(sqlParser.T__2)
                pass
            elif la_ == 5:
                localctx = sqlParser.StructContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2473
                self.match(sqlParser.STRUCT)
                self.state = 2474
                self.match(sqlParser.T__1)
                self.state = 2483
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 319, self._ctx)
                if la_ == 1:
                    self.state = 2475
                    localctx._namedExpression = self.namedExpression()
                    localctx.argument.append(localctx._namedExpression)
                    self.state = 2480
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 2476
                        self.match(sqlParser.T__3)
                        self.state = 2477
                        localctx._namedExpression = self.namedExpression()
                        localctx.argument.append(localctx._namedExpression)
                        self.state = 2482
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 2485
                self.match(sqlParser.T__2)
                pass
            elif la_ == 6:
                localctx = sqlParser.FirstContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2486
                self.match(sqlParser.FIRST)
                self.state = 2487
                self.match(sqlParser.T__1)
                self.state = 2488
                self.expression()
                self.state = 2491
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 111:
                    self.state = 2489
                    self.match(sqlParser.IGNORE)
                    self.state = 2490
                    self.match(sqlParser.NULLS)
                self.state = 2493
                self.match(sqlParser.T__2)
                pass
            elif la_ == 7:
                localctx = sqlParser.LastContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2495
                self.match(sqlParser.LAST)
                self.state = 2496
                self.match(sqlParser.T__1)
                self.state = 2497
                self.expression()
                self.state = 2500
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 111:
                    self.state = 2498
                    self.match(sqlParser.IGNORE)
                    self.state = 2499
                    self.match(sqlParser.NULLS)
                self.state = 2502
                self.match(sqlParser.T__2)
                pass
            elif la_ == 8:
                localctx = sqlParser.PositionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2504
                self.match(sqlParser.POSITION)
                self.state = 2505
                self.match(sqlParser.T__1)
                self.state = 2506
                localctx.substr = self.valueExpression(0)
                self.state = 2507
                self.match(sqlParser.IN)
                self.state = 2508
                localctx.qpdstr = self.valueExpression(0)
                self.state = 2509
                self.match(sqlParser.T__2)
                pass
            elif la_ == 9:
                localctx = sqlParser.ConstantDefaultContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2511
                self.constant()
                pass
            elif la_ == 10:
                localctx = sqlParser.StarContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2512
                self.match(sqlParser.ASTERISK)
                pass
            elif la_ == 11:
                localctx = sqlParser.StarContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2513
                self.qualifiedName()
                self.state = 2514
                self.match(sqlParser.T__4)
                self.state = 2515
                self.match(sqlParser.ASTERISK)
                pass
            elif la_ == 12:
                localctx = sqlParser.RowConstructorContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2517
                self.match(sqlParser.T__1)
                self.state = 2518
                self.namedExpression()
                self.state = 2521
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 2519
                    self.match(sqlParser.T__3)
                    self.state = 2520
                    self.namedExpression()
                    self.state = 2523
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 4:
                        break
                self.state = 2525
                self.match(sqlParser.T__2)
                pass
            elif la_ == 13:
                localctx = sqlParser.SubqueryExpressionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2527
                self.match(sqlParser.T__1)
                self.state = 2528
                self.query()
                self.state = 2529
                self.match(sqlParser.T__2)
                pass
            elif la_ == 14:
                localctx = sqlParser.FunctionCallContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2531
                self.functionName()
                self.state = 2532
                self.match(sqlParser.T__1)
                self.state = 2544
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 325, self._ctx)
                if la_ == 1:
                    self.state = 2534
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 323, self._ctx)
                    if la_ == 1:
                        self.state = 2533
                        self.setQuantifier()
                    self.state = 2536
                    localctx._expression = self.expression()
                    localctx.argument.append(localctx._expression)
                    self.state = 2541
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 2537
                        self.match(sqlParser.T__3)
                        self.state = 2538
                        localctx._expression = self.expression()
                        localctx.argument.append(localctx._expression)
                        self.state = 2543
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 2546
                self.match(sqlParser.T__2)
                self.state = 2553
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 326, self._ctx)
                if la_ == 1:
                    self.state = 2547
                    self.match(sqlParser.FILTER)
                    self.state = 2548
                    self.match(sqlParser.T__1)
                    self.state = 2549
                    self.match(sqlParser.WHERE)
                    self.state = 2550
                    localctx.where = self.booleanExpression(0)
                    self.state = 2551
                    self.match(sqlParser.T__2)
                self.state = 2557
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 327, self._ctx)
                if la_ == 1:
                    self.state = 2555
                    self.match(sqlParser.OVER)
                    self.state = 2556
                    self.windowSpec()
                pass
            elif la_ == 15:
                localctx = sqlParser.LambdaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2559
                self.identifier()
                self.state = 2560
                self.match(sqlParser.T__7)
                self.state = 2561
                self.expression()
                pass
            elif la_ == 16:
                localctx = sqlParser.LambdaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2563
                self.match(sqlParser.T__1)
                self.state = 2564
                self.identifier()
                self.state = 2567
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 2565
                    self.match(sqlParser.T__3)
                    self.state = 2566
                    self.identifier()
                    self.state = 2569
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 4:
                        break
                self.state = 2571
                self.match(sqlParser.T__2)
                self.state = 2572
                self.match(sqlParser.T__7)
                self.state = 2573
                self.expression()
                pass
            elif la_ == 17:
                localctx = sqlParser.ColumnReferenceContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2575
                self.identifier()
                pass
            elif la_ == 18:
                localctx = sqlParser.ParenthesizedExpressionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2576
                self.match(sqlParser.T__1)
                self.state = 2577
                self.expression()
                self.state = 2578
                self.match(sqlParser.T__2)
                pass
            elif la_ == 19:
                localctx = sqlParser.ExtractContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2580
                self.match(sqlParser.EXTRACT)
                self.state = 2581
                self.match(sqlParser.T__1)
                self.state = 2582
                localctx.field = self.identifier()
                self.state = 2583
                self.match(sqlParser.FROM)
                self.state = 2584
                localctx.source = self.valueExpression(0)
                self.state = 2585
                self.match(sqlParser.T__2)
                pass
            elif la_ == 20:
                localctx = sqlParser.SubstringContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2587
                _la = self._input.LA(1)
                if not (_la == 225 or _la == 226):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2588
                self.match(sqlParser.T__1)
                self.state = 2589
                localctx.qpdstr = self.valueExpression(0)
                self.state = 2590
                _la = self._input.LA(1)
                if not (_la == 4 or _la == 100):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2591
                localctx.pos = self.valueExpression(0)
                self.state = 2594
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 4 or _la == 96:
                    self.state = 2592
                    _la = self._input.LA(1)
                    if not (_la == 4 or _la == 96):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 2593
                    localctx.qpdlen = self.valueExpression(0)
                self.state = 2596
                self.match(sqlParser.T__2)
                pass
            elif la_ == 21:
                localctx = sqlParser.TrimContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2598
                self.match(sqlParser.TRIM)
                self.state = 2599
                self.match(sqlParser.T__1)
                self.state = 2601
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 330, self._ctx)
                if la_ == 1:
                    self.state = 2600
                    localctx.trimOption = self._input.LT(1)
                    _la = self._input.LA(1)
                    if not (_la == 27 or _la == 130 or _la == 236):
                        localctx.trimOption = self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 2604
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 331, self._ctx)
                if la_ == 1:
                    self.state = 2603
                    localctx.trimStr = self.valueExpression(0)
                self.state = 2606
                self.match(sqlParser.FROM)
                self.state = 2607
                localctx.srcStr = self.valueExpression(0)
                self.state = 2608
                self.match(sqlParser.T__2)
                pass
            elif la_ == 22:
                localctx = sqlParser.OverlayContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2610
                self.match(sqlParser.OVERLAY)
                self.state = 2611
                self.match(sqlParser.T__1)
                self.state = 2612
                localctx.qpdinput = self.valueExpression(0)
                self.state = 2613
                self.match(sqlParser.PLACING)
                self.state = 2614
                localctx.replace = self.valueExpression(0)
                self.state = 2615
                self.match(sqlParser.FROM)
                self.state = 2616
                localctx.position = self.valueExpression(0)
                self.state = 2619
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 96:
                    self.state = 2617
                    self.match(sqlParser.FOR)
                    self.state = 2618
                    localctx.length = self.valueExpression(0)
                self.state = 2621
                self.match(sqlParser.T__2)
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 2635
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 335, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 2633
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 334, self._ctx)
                    if la_ == 1:
                        localctx = sqlParser.SubscriptContext(self, sqlParser.PrimaryExpressionContext(self, _parentctx, _parentState))
                        localctx.value = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_primaryExpression)
                        self.state = 2625
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 8)')
                        self.state = 2626
                        self.match(sqlParser.T__8)
                        self.state = 2627
                        localctx.index = self.valueExpression(0)
                        self.state = 2628
                        self.match(sqlParser.T__9)
                        pass
                    elif la_ == 2:
                        localctx = sqlParser.DereferenceContext(self, sqlParser.PrimaryExpressionContext(self, _parentctx, _parentState))
                        localctx.base = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_primaryExpression)
                        self.state = 2630
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 6)')
                        self.state = 2631
                        self.match(sqlParser.T__4)
                        self.state = 2632
                        localctx.fieldName = self.identifier()
                        pass
                self.state = 2637
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 335, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class ConstantContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_constant

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class NullLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNullLiteral'):
                return visitor.visitNullLiteral(self)
            else:
                return visitor.visitChildren(self)

    class StringLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.STRING)
            else:
                return self.getToken(sqlParser.STRING, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStringLiteral'):
                return visitor.visitStringLiteral(self)
            else:
                return visitor.visitChildren(self)

    class TypeConstructorContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTypeConstructor'):
                return visitor.visitTypeConstructor(self)
            else:
                return visitor.visitChildren(self)

    class IntervalLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def interval(self):
            return self.getTypedRuleContext(sqlParser.IntervalContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntervalLiteral'):
                return visitor.visitIntervalLiteral(self)
            else:
                return visitor.visitChildren(self)

    class NumericLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def number(self):
            return self.getTypedRuleContext(sqlParser.NumberContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNumericLiteral'):
                return visitor.visitNumericLiteral(self)
            else:
                return visitor.visitChildren(self)

    class BooleanLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def booleanValue(self):
            return self.getTypedRuleContext(sqlParser.BooleanValueContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBooleanLiteral'):
                return visitor.visitBooleanLiteral(self)
            else:
                return visitor.visitChildren(self)

    def constant(self):
        localctx = sqlParser.ConstantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 196, self.RULE_constant)
        try:
            self.state = 2650
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 337, self._ctx)
            if la_ == 1:
                localctx = sqlParser.NullLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2638
                self.match(sqlParser.NULL)
                pass
            elif la_ == 2:
                localctx = sqlParser.IntervalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2639
                self.interval()
                pass
            elif la_ == 3:
                localctx = sqlParser.TypeConstructorContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2640
                self.identifier()
                self.state = 2641
                self.match(sqlParser.STRING)
                pass
            elif la_ == 4:
                localctx = sqlParser.NumericLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2643
                self.number()
                pass
            elif la_ == 5:
                localctx = sqlParser.BooleanLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 2644
                self.booleanValue()
                pass
            elif la_ == 6:
                localctx = sqlParser.StringLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 2646
                self._errHandler.sync(self)
                _alt = 1
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2645
                        self.match(sqlParser.STRING)
                    else:
                        raise NoViableAltException(self)
                    self.state = 2648
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 336, self._ctx)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ComparisonOperatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EQ(self):
            return self.getToken(sqlParser.EQ, 0)

        def NEQ(self):
            return self.getToken(sqlParser.NEQ, 0)

        def NEQJ(self):
            return self.getToken(sqlParser.NEQJ, 0)

        def LT(self):
            return self.getToken(sqlParser.LT, 0)

        def LTE(self):
            return self.getToken(sqlParser.LTE, 0)

        def GT(self):
            return self.getToken(sqlParser.GT, 0)

        def GTE(self):
            return self.getToken(sqlParser.GTE, 0)

        def NSEQ(self):
            return self.getToken(sqlParser.NSEQ, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_comparisonOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComparisonOperator'):
                return visitor.visitComparisonOperator(self)
            else:
                return visitor.visitChildren(self)

    def comparisonOperator(self):
        localctx = sqlParser.ComparisonOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 198, self.RULE_comparisonOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2652
            _la = self._input.LA(1)
            if not (_la - 264 & ~63 == 0 and 1 << _la - 264 & 255 != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ArithmeticOperatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PLUS(self):
            return self.getToken(sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def ASTERISK(self):
            return self.getToken(sqlParser.ASTERISK, 0)

        def SLASH(self):
            return self.getToken(sqlParser.SLASH, 0)

        def PERCENT(self):
            return self.getToken(sqlParser.PERCENT, 0)

        def DIV(self):
            return self.getToken(sqlParser.DIV, 0)

        def TILDE(self):
            return self.getToken(sqlParser.TILDE, 0)

        def AMPERSAND(self):
            return self.getToken(sqlParser.AMPERSAND, 0)

        def PIPE(self):
            return self.getToken(sqlParser.PIPE, 0)

        def CONCAT_PIPE(self):
            return self.getToken(sqlParser.CONCAT_PIPE, 0)

        def HAT(self):
            return self.getToken(sqlParser.HAT, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_arithmeticOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticOperator'):
                return visitor.visitArithmeticOperator(self)
            else:
                return visitor.visitChildren(self)

    def arithmeticOperator(self):
        localctx = sqlParser.ArithmeticOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 200, self.RULE_arithmeticOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2654
            _la = self._input.LA(1)
            if not (_la - 272 & ~63 == 0 and 1 << _la - 272 & 2047 != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PredicateOperatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_predicateOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPredicateOperator'):
                return visitor.visitPredicateOperator(self)
            else:
                return visitor.visitChildren(self)

    def predicateOperator(self):
        localctx = sqlParser.PredicateOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 202, self.RULE_predicateOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2656
            _la = self._input.LA(1)
            if not (_la == 17 or (_la - 113 & ~63 == 0 and 1 << _la - 113 & 282574488338433 != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class BooleanValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRUE(self):
            return self.getToken(sqlParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(sqlParser.FALSE, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_booleanValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBooleanValue'):
                return visitor.visitBooleanValue(self)
            else:
                return visitor.visitChildren(self)

    def booleanValue(self):
        localctx = sqlParser.BooleanValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 204, self.RULE_booleanValue)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2658
            _la = self._input.LA(1)
            if not (_la == 89 or _la == 241):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INTERVAL(self):
            return self.getToken(sqlParser.INTERVAL, 0)

        def errorCapturingMultiUnitsInterval(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingMultiUnitsIntervalContext, 0)

        def errorCapturingUnitToUnitInterval(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingUnitToUnitIntervalContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_interval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInterval'):
                return visitor.visitInterval(self)
            else:
                return visitor.visitChildren(self)

    def interval(self):
        localctx = sqlParser.IntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 206, self.RULE_interval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2660
            self.match(sqlParser.INTERVAL)
            self.state = 2663
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 338, self._ctx)
            if la_ == 1:
                self.state = 2661
                self.errorCapturingMultiUnitsInterval()
            elif la_ == 2:
                self.state = 2662
                self.errorCapturingUnitToUnitInterval()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ErrorCapturingMultiUnitsIntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def multiUnitsInterval(self):
            return self.getTypedRuleContext(sqlParser.MultiUnitsIntervalContext, 0)

        def unitToUnitInterval(self):
            return self.getTypedRuleContext(sqlParser.UnitToUnitIntervalContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_errorCapturingMultiUnitsInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingMultiUnitsInterval'):
                return visitor.visitErrorCapturingMultiUnitsInterval(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingMultiUnitsInterval(self):
        localctx = sqlParser.ErrorCapturingMultiUnitsIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 208, self.RULE_errorCapturingMultiUnitsInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2665
            self.multiUnitsInterval()
            self.state = 2667
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 339, self._ctx)
            if la_ == 1:
                self.state = 2666
                self.unitToUnitInterval()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class MultiUnitsIntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def intervalValue(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IntervalValueContext)
            else:
                return self.getTypedRuleContext(sqlParser.IntervalValueContext, i)

        def intervalUnit(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IntervalUnitContext)
            else:
                return self.getTypedRuleContext(sqlParser.IntervalUnitContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_multiUnitsInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultiUnitsInterval'):
                return visitor.visitMultiUnitsInterval(self)
            else:
                return visitor.visitChildren(self)

    def multiUnitsInterval(self):
        localctx = sqlParser.MultiUnitsIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 210, self.RULE_multiUnitsInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2672
            self._errHandler.sync(self)
            _alt = 1
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2669
                    self.intervalValue()
                    self.state = 2670
                    self.intervalUnit()
                else:
                    raise NoViableAltException(self)
                self.state = 2674
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 340, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ErrorCapturingUnitToUnitIntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.body = None
            self.error1 = None
            self.error2 = None

        def unitToUnitInterval(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.UnitToUnitIntervalContext)
            else:
                return self.getTypedRuleContext(sqlParser.UnitToUnitIntervalContext, i)

        def multiUnitsInterval(self):
            return self.getTypedRuleContext(sqlParser.MultiUnitsIntervalContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_errorCapturingUnitToUnitInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingUnitToUnitInterval'):
                return visitor.visitErrorCapturingUnitToUnitInterval(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingUnitToUnitInterval(self):
        localctx = sqlParser.ErrorCapturingUnitToUnitIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 212, self.RULE_errorCapturingUnitToUnitInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2676
            localctx.body = self.unitToUnitInterval()
            self.state = 2679
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 341, self._ctx)
            if la_ == 1:
                self.state = 2677
                localctx.error1 = self.multiUnitsInterval()
            elif la_ == 2:
                self.state = 2678
                localctx.error2 = self.unitToUnitInterval()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class UnitToUnitIntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.value = None
            self.qpdfrom = None
            self.to = None

        def TO(self):
            return self.getToken(sqlParser.TO, 0)

        def intervalValue(self):
            return self.getTypedRuleContext(sqlParser.IntervalValueContext, 0)

        def intervalUnit(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IntervalUnitContext)
            else:
                return self.getTypedRuleContext(sqlParser.IntervalUnitContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_unitToUnitInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnitToUnitInterval'):
                return visitor.visitUnitToUnitInterval(self)
            else:
                return visitor.visitChildren(self)

    def unitToUnitInterval(self):
        localctx = sqlParser.UnitToUnitIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 214, self.RULE_unitToUnitInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2681
            localctx.value = self.intervalValue()
            self.state = 2682
            localctx.qpdfrom = self.intervalUnit()
            self.state = 2683
            self.match(sqlParser.TO)
            self.state = 2684
            localctx.to = self.intervalUnit()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IntervalValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INTEGER_VALUE(self):
            return self.getToken(sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(sqlParser.DECIMAL_VALUE, 0)

        def PLUS(self):
            return self.getToken(sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def STRING(self):
            return self.getToken(sqlParser.STRING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_intervalValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntervalValue'):
                return visitor.visitIntervalValue(self)
            else:
                return visitor.visitChildren(self)

    def intervalValue(self):
        localctx = sqlParser.IntervalValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 216, self.RULE_intervalValue)
        self._la = 0
        try:
            self.state = 2691
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [272, 273, 287, 289]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2687
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 272 or _la == 273:
                    self.state = 2686
                    _la = self._input.LA(1)
                    if not (_la == 272 or _la == 273):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 2689
                _la = self._input.LA(1)
                if not (_la == 287 or _la == 289):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif token in [283]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2690
                self.match(sqlParser.STRING)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IntervalUnitContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DAY(self):
            return self.getToken(sqlParser.DAY, 0)

        def HOUR(self):
            return self.getToken(sqlParser.HOUR, 0)

        def MINUTE(self):
            return self.getToken(sqlParser.MINUTE, 0)

        def MONTH(self):
            return self.getToken(sqlParser.MONTH, 0)

        def SECOND(self):
            return self.getToken(sqlParser.SECOND, 0)

        def YEAR(self):
            return self.getToken(sqlParser.YEAR, 0)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_intervalUnit

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntervalUnit'):
                return visitor.visitIntervalUnit(self)
            else:
                return visitor.visitChildren(self)

    def intervalUnit(self):
        localctx = sqlParser.IntervalUnitContext(self, self._ctx, self.state)
        self.enterRule(localctx, 218, self.RULE_intervalUnit)
        try:
            self.state = 2700
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 344, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2693
                self.match(sqlParser.DAY)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2694
                self.match(sqlParser.HOUR)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2695
                self.match(sqlParser.MINUTE)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2696
                self.match(sqlParser.MONTH)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 2697
                self.match(sqlParser.SECOND)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 2698
                self.match(sqlParser.YEAR)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 2699
                self.identifier()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ColPositionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.position = None
            self.afterCol = None

        def FIRST(self):
            return self.getToken(sqlParser.FIRST, 0)

        def AFTER(self):
            return self.getToken(sqlParser.AFTER, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_colPosition

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColPosition'):
                return visitor.visitColPosition(self)
            else:
                return visitor.visitChildren(self)

    def colPosition(self):
        localctx = sqlParser.ColPositionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 220, self.RULE_colPosition)
        try:
            self.state = 2705
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [94]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2702
                localctx.position = self.match(sqlParser.FIRST)
                pass
            elif token in [13]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2703
                localctx.position = self.match(sqlParser.AFTER)
                self.state = 2704
                localctx.afterCol = self.errorCapturingIdentifier()
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DataTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_dataType

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ComplexDataTypeContext(DataTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.qpdcomplex = None
            self.copyFrom(ctx)

        def LT(self):
            return self.getToken(sqlParser.LT, 0)

        def dataType(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.DataTypeContext)
            else:
                return self.getTypedRuleContext(sqlParser.DataTypeContext, i)

        def GT(self):
            return self.getToken(sqlParser.GT, 0)

        def ARRAY(self):
            return self.getToken(sqlParser.ARRAY, 0)

        def MAP(self):
            return self.getToken(sqlParser.MAP, 0)

        def STRUCT(self):
            return self.getToken(sqlParser.STRUCT, 0)

        def NEQ(self):
            return self.getToken(sqlParser.NEQ, 0)

        def complexColTypeList(self):
            return self.getTypedRuleContext(sqlParser.ComplexColTypeListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComplexDataType'):
                return visitor.visitComplexDataType(self)
            else:
                return visitor.visitChildren(self)

    class PrimitiveDataTypeContext(DataTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def INTEGER_VALUE(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.INTEGER_VALUE)
            else:
                return self.getToken(sqlParser.INTEGER_VALUE, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPrimitiveDataType'):
                return visitor.visitPrimitiveDataType(self)
            else:
                return visitor.visitChildren(self)

    def dataType(self):
        localctx = sqlParser.DataTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 222, self.RULE_dataType)
        self._la = 0
        try:
            self.state = 2741
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 350, self._ctx)
            if la_ == 1:
                localctx = sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2707
                localctx.qpdcomplex = self.match(sqlParser.ARRAY)
                self.state = 2708
                self.match(sqlParser.LT)
                self.state = 2709
                self.dataType()
                self.state = 2710
                self.match(sqlParser.GT)
                pass
            elif la_ == 2:
                localctx = sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2712
                localctx.qpdcomplex = self.match(sqlParser.MAP)
                self.state = 2713
                self.match(sqlParser.LT)
                self.state = 2714
                self.dataType()
                self.state = 2715
                self.match(sqlParser.T__3)
                self.state = 2716
                self.dataType()
                self.state = 2717
                self.match(sqlParser.GT)
                pass
            elif la_ == 3:
                localctx = sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2719
                localctx.qpdcomplex = self.match(sqlParser.STRUCT)
                self.state = 2726
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [268]:
                    self.state = 2720
                    self.match(sqlParser.LT)
                    self.state = 2722
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 346, self._ctx)
                    if la_ == 1:
                        self.state = 2721
                        self.complexColTypeList()
                    self.state = 2724
                    self.match(sqlParser.GT)
                    pass
                elif token in [266]:
                    self.state = 2725
                    self.match(sqlParser.NEQ)
                    pass
                else:
                    raise NoViableAltException(self)
                pass
            elif la_ == 4:
                localctx = sqlParser.PrimitiveDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2728
                self.identifier()
                self.state = 2739
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 349, self._ctx)
                if la_ == 1:
                    self.state = 2729
                    self.match(sqlParser.T__1)
                    self.state = 2730
                    self.match(sqlParser.INTEGER_VALUE)
                    self.state = 2735
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 2731
                        self.match(sqlParser.T__3)
                        self.state = 2732
                        self.match(sqlParser.INTEGER_VALUE)
                        self.state = 2737
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 2738
                    self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QualifiedColTypeWithPositionListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def qualifiedColTypeWithPosition(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.QualifiedColTypeWithPositionContext)
            else:
                return self.getTypedRuleContext(sqlParser.QualifiedColTypeWithPositionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_qualifiedColTypeWithPositionList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedColTypeWithPositionList'):
                return visitor.visitQualifiedColTypeWithPositionList(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedColTypeWithPositionList(self):
        localctx = sqlParser.QualifiedColTypeWithPositionListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 224, self.RULE_qualifiedColTypeWithPositionList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2743
            self.qualifiedColTypeWithPosition()
            self.state = 2748
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2744
                self.match(sqlParser.T__3)
                self.state = 2745
                self.qualifiedColTypeWithPosition()
                self.state = 2750
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QualifiedColTypeWithPositionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(sqlParser.MultipartIdentifierContext, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(sqlParser.CommentSpecContext, 0)

        def colPosition(self):
            return self.getTypedRuleContext(sqlParser.ColPositionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_qualifiedColTypeWithPosition

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedColTypeWithPosition'):
                return visitor.visitQualifiedColTypeWithPosition(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedColTypeWithPosition(self):
        localctx = sqlParser.QualifiedColTypeWithPositionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 226, self.RULE_qualifiedColTypeWithPosition)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2751
            localctx.name = self.multipartIdentifier()
            self.state = 2752
            self.dataType()
            self.state = 2755
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 153:
                self.state = 2753
                self.match(sqlParser.NOT)
                self.state = 2754
                self.match(sqlParser.NULL)
            self.state = 2758
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 45:
                self.state = 2757
                self.commentSpec()
            self.state = 2761
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 13 or _la == 94:
                self.state = 2760
                self.colPosition()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ColTypeListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def colType(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ColTypeContext)
            else:
                return self.getTypedRuleContext(sqlParser.ColTypeContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_colTypeList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColTypeList'):
                return visitor.visitColTypeList(self)
            else:
                return visitor.visitChildren(self)

    def colTypeList(self):
        localctx = sqlParser.ColTypeListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 228, self.RULE_colTypeList)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2763
            self.colType()
            self.state = 2768
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 355, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2764
                    self.match(sqlParser.T__3)
                    self.state = 2765
                    self.colType()
                self.state = 2770
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 355, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ColTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.colName = None

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_colType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColType'):
                return visitor.visitColType(self)
            else:
                return visitor.visitChildren(self)

    def colType(self):
        localctx = sqlParser.ColTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 230, self.RULE_colType)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2771
            localctx.colName = self.errorCapturingIdentifier()
            self.state = 2772
            self.dataType()
            self.state = 2775
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 356, self._ctx)
            if la_ == 1:
                self.state = 2773
                self.match(sqlParser.NOT)
                self.state = 2774
                self.match(sqlParser.NULL)
            self.state = 2778
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 357, self._ctx)
            if la_ == 1:
                self.state = 2777
                self.commentSpec()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ComplexColTypeListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def complexColType(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ComplexColTypeContext)
            else:
                return self.getTypedRuleContext(sqlParser.ComplexColTypeContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_complexColTypeList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComplexColTypeList'):
                return visitor.visitComplexColTypeList(self)
            else:
                return visitor.visitChildren(self)

    def complexColTypeList(self):
        localctx = sqlParser.ComplexColTypeListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 232, self.RULE_complexColTypeList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2780
            self.complexColType()
            self.state = 2785
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2781
                self.match(sqlParser.T__3)
                self.state = 2782
                self.complexColType()
                self.state = 2787
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ComplexColTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_complexColType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComplexColType'):
                return visitor.visitComplexColType(self)
            else:
                return visitor.visitChildren(self)

    def complexColType(self):
        localctx = sqlParser.ComplexColTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 234, self.RULE_complexColType)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2788
            self.identifier()
            self.state = 2789
            self.match(sqlParser.T__10)
            self.state = 2790
            self.dataType()
            self.state = 2793
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 153:
                self.state = 2791
                self.match(sqlParser.NOT)
                self.state = 2792
                self.match(sqlParser.NULL)
            self.state = 2796
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 45:
                self.state = 2795
                self.commentSpec()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WhenClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.condition = None
            self.result = None

        def WHEN(self):
            return self.getToken(sqlParser.WHEN, 0)

        def THEN(self):
            return self.getToken(sqlParser.THEN, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_whenClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWhenClause'):
                return visitor.visitWhenClause(self)
            else:
                return visitor.visitChildren(self)

    def whenClause(self):
        localctx = sqlParser.WhenClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 236, self.RULE_whenClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2798
            self.match(sqlParser.WHEN)
            self.state = 2799
            localctx.condition = self.expression()
            self.state = 2800
            self.match(sqlParser.THEN)
            self.state = 2801
            localctx.result = self.expression()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WindowClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def WINDOW(self):
            return self.getToken(sqlParser.WINDOW, 0)

        def namedWindow(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.NamedWindowContext)
            else:
                return self.getTypedRuleContext(sqlParser.NamedWindowContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_windowClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowClause'):
                return visitor.visitWindowClause(self)
            else:
                return visitor.visitChildren(self)

    def windowClause(self):
        localctx = sqlParser.WindowClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 238, self.RULE_windowClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2803
            self.match(sqlParser.WINDOW)
            self.state = 2804
            self.namedWindow()
            self.state = 2809
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 361, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2805
                    self.match(sqlParser.T__3)
                    self.state = 2806
                    self.namedWindow()
                self.state = 2811
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 361, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NamedWindowContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def windowSpec(self):
            return self.getTypedRuleContext(sqlParser.WindowSpecContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_namedWindow

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedWindow'):
                return visitor.visitNamedWindow(self)
            else:
                return visitor.visitChildren(self)

    def namedWindow(self):
        localctx = sqlParser.NamedWindowContext(self, self._ctx, self.state)
        self.enterRule(localctx, 240, self.RULE_namedWindow)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2812
            localctx.name = self.errorCapturingIdentifier()
            self.state = 2813
            self.match(sqlParser.AS)
            self.state = 2814
            self.windowSpec()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WindowSpecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_windowSpec

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class WindowRefContext(WindowSpecContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.name = None
            self.copyFrom(ctx)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowRef'):
                return visitor.visitWindowRef(self)
            else:
                return visitor.visitChildren(self)

    class WindowDefContext(WindowSpecContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self._expression = None
            self.partition = list()
            self.copyFrom(ctx)

        def CLUSTER(self):
            return self.getToken(sqlParser.CLUSTER, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.BY)
            else:
                return self.getToken(sqlParser.BY, i)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(sqlParser.ExpressionContext, i)

        def windowFrame(self):
            return self.getTypedRuleContext(sqlParser.WindowFrameContext, 0)

        def sortItem(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.SortItemContext)
            else:
                return self.getTypedRuleContext(sqlParser.SortItemContext, i)

        def PARTITION(self):
            return self.getToken(sqlParser.PARTITION, 0)

        def DISTRIBUTE(self):
            return self.getToken(sqlParser.DISTRIBUTE, 0)

        def ORDER(self):
            return self.getToken(sqlParser.ORDER, 0)

        def SORT(self):
            return self.getToken(sqlParser.SORT, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowDef'):
                return visitor.visitWindowDef(self)
            else:
                return visitor.visitChildren(self)

    def windowSpec(self):
        localctx = sqlParser.WindowSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 242, self.RULE_windowSpec)
        self._la = 0
        try:
            self.state = 2862
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 369, self._ctx)
            if la_ == 1:
                localctx = sqlParser.WindowRefContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2816
                localctx.name = self.errorCapturingIdentifier()
                pass
            elif la_ == 2:
                localctx = sqlParser.WindowRefContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2817
                self.match(sqlParser.T__1)
                self.state = 2818
                localctx.name = self.errorCapturingIdentifier()
                self.state = 2819
                self.match(sqlParser.T__2)
                pass
            elif la_ == 3:
                localctx = sqlParser.WindowDefContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2821
                self.match(sqlParser.T__1)
                self.state = 2856
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [38]:
                    self.state = 2822
                    self.match(sqlParser.CLUSTER)
                    self.state = 2823
                    self.match(sqlParser.BY)
                    self.state = 2824
                    localctx._expression = self.expression()
                    localctx.partition.append(localctx._expression)
                    self.state = 2829
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 4:
                        self.state = 2825
                        self.match(sqlParser.T__3)
                        self.state = 2826
                        localctx._expression = self.expression()
                        localctx.partition.append(localctx._expression)
                        self.state = 2831
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    pass
                elif token in [3, 75, 162, 170, 183, 203, 218]:
                    self.state = 2842
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 75 or _la == 170:
                        self.state = 2832
                        _la = self._input.LA(1)
                        if not (_la == 75 or _la == 170):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2833
                        self.match(sqlParser.BY)
                        self.state = 2834
                        localctx._expression = self.expression()
                        localctx.partition.append(localctx._expression)
                        self.state = 2839
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        while _la == 4:
                            self.state = 2835
                            self.match(sqlParser.T__3)
                            self.state = 2836
                            localctx._expression = self.expression()
                            localctx.partition.append(localctx._expression)
                            self.state = 2841
                            self._errHandler.sync(self)
                            _la = self._input.LA(1)
                    self.state = 2854
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 162 or _la == 218:
                        self.state = 2844
                        _la = self._input.LA(1)
                        if not (_la == 162 or _la == 218):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2845
                        self.match(sqlParser.BY)
                        self.state = 2846
                        self.sortItem()
                        self.state = 2851
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        while _la == 4:
                            self.state = 2847
                            self.match(sqlParser.T__3)
                            self.state = 2848
                            self.sortItem()
                            self.state = 2853
                            self._errHandler.sync(self)
                            _la = self._input.LA(1)
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 2859
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 183 or _la == 203:
                    self.state = 2858
                    self.windowFrame()
                self.state = 2861
                self.match(sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class WindowFrameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.frameType = None
            self.start = None
            self.end = None

        def RANGE(self):
            return self.getToken(sqlParser.RANGE, 0)

        def frameBound(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.FrameBoundContext)
            else:
                return self.getTypedRuleContext(sqlParser.FrameBoundContext, i)

        def ROWS(self):
            return self.getToken(sqlParser.ROWS, 0)

        def BETWEEN(self):
            return self.getToken(sqlParser.BETWEEN, 0)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_windowFrame

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowFrame'):
                return visitor.visitWindowFrame(self)
            else:
                return visitor.visitChildren(self)

    def windowFrame(self):
        localctx = sqlParser.WindowFrameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 244, self.RULE_windowFrame)
        try:
            self.state = 2880
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 370, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2864
                localctx.frameType = self.match(sqlParser.RANGE)
                self.state = 2865
                localctx.start = self.frameBound()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2866
                localctx.frameType = self.match(sqlParser.ROWS)
                self.state = 2867
                localctx.start = self.frameBound()
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2868
                localctx.frameType = self.match(sqlParser.RANGE)
                self.state = 2869
                self.match(sqlParser.BETWEEN)
                self.state = 2870
                localctx.start = self.frameBound()
                self.state = 2871
                self.match(sqlParser.AND)
                self.state = 2872
                localctx.end = self.frameBound()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2874
                localctx.frameType = self.match(sqlParser.ROWS)
                self.state = 2875
                self.match(sqlParser.BETWEEN)
                self.state = 2876
                localctx.start = self.frameBound()
                self.state = 2877
                self.match(sqlParser.AND)
                self.state = 2878
                localctx.end = self.frameBound()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FrameBoundContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.boundType = None

        def UNBOUNDED(self):
            return self.getToken(sqlParser.UNBOUNDED, 0)

        def PRECEDING(self):
            return self.getToken(sqlParser.PRECEDING, 0)

        def FOLLOWING(self):
            return self.getToken(sqlParser.FOLLOWING, 0)

        def ROW(self):
            return self.getToken(sqlParser.ROW, 0)

        def CURRENT(self):
            return self.getToken(sqlParser.CURRENT, 0)

        def expression(self):
            return self.getTypedRuleContext(sqlParser.ExpressionContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_frameBound

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFrameBound'):
                return visitor.visitFrameBound(self)
            else:
                return visitor.visitChildren(self)

    def frameBound(self):
        localctx = sqlParser.FrameBoundContext(self, self._ctx, self.state)
        self.enterRule(localctx, 246, self.RULE_frameBound)
        self._la = 0
        try:
            self.state = 2889
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 371, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2882
                self.match(sqlParser.UNBOUNDED)
                self.state = 2883
                localctx.boundType = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 95 or _la == 177):
                    localctx.boundType = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2884
                localctx.boundType = self.match(sqlParser.CURRENT)
                self.state = 2885
                self.match(sqlParser.ROW)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2886
                self.expression()
                self.state = 2887
                localctx.boundType = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 95 or _la == 177):
                    localctx.boundType = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QualifiedNameListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def qualifiedName(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.QualifiedNameContext)
            else:
                return self.getTypedRuleContext(sqlParser.QualifiedNameContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_qualifiedNameList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedNameList'):
                return visitor.visitQualifiedNameList(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedNameList(self):
        localctx = sqlParser.QualifiedNameListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 248, self.RULE_qualifiedNameList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2891
            self.qualifiedName()
            self.state = 2896
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 4:
                self.state = 2892
                self.match(sqlParser.T__3)
                self.state = 2893
                self.qualifiedName()
                self.state = 2898
                self._errHandler.sync(self)
                _la = self._input.LA(1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FunctionNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def qualifiedName(self):
            return self.getTypedRuleContext(sqlParser.QualifiedNameContext, 0)

        def FILTER(self):
            return self.getToken(sqlParser.FILTER, 0)

        def LEFT(self):
            return self.getToken(sqlParser.LEFT, 0)

        def RIGHT(self):
            return self.getToken(sqlParser.RIGHT, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_functionName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionName'):
                return visitor.visitFunctionName(self)
            else:
                return visitor.visitChildren(self)

    def functionName(self):
        localctx = sqlParser.FunctionNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 250, self.RULE_functionName)
        try:
            self.state = 2903
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 373, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2899
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2900
                self.match(sqlParser.FILTER)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2901
                self.match(sqlParser.LEFT)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2902
                self.match(sqlParser.RIGHT)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QualifiedNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return sqlParser.RULE_qualifiedName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedName'):
                return visitor.visitQualifiedName(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedName(self):
        localctx = sqlParser.QualifiedNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 252, self.RULE_qualifiedName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2905
            self.identifier()
            self.state = 2910
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 374, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2906
                    self.match(sqlParser.T__4)
                    self.state = 2907
                    self.identifier()
                self.state = 2912
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 374, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ErrorCapturingIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(sqlParser.IdentifierContext, 0)

        def errorCapturingIdentifierExtra(self):
            return self.getTypedRuleContext(sqlParser.ErrorCapturingIdentifierExtraContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_errorCapturingIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingIdentifier'):
                return visitor.visitErrorCapturingIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingIdentifier(self):
        localctx = sqlParser.ErrorCapturingIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 254, self.RULE_errorCapturingIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2913
            self.identifier()
            self.state = 2914
            self.errorCapturingIdentifierExtra()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ErrorCapturingIdentifierExtraContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_errorCapturingIdentifierExtra

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ErrorIdentContext(ErrorCapturingIdentifierExtraContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def MINUS(self, i: int=None):
            if i is None:
                return self.getTokens(sqlParser.MINUS)
            else:
                return self.getToken(sqlParser.MINUS, i)

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(sqlParser.IdentifierContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorIdent'):
                return visitor.visitErrorIdent(self)
            else:
                return visitor.visitChildren(self)

    class RealIdentContext(ErrorCapturingIdentifierExtraContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRealIdent'):
                return visitor.visitRealIdent(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingIdentifierExtra(self):
        localctx = sqlParser.ErrorCapturingIdentifierExtraContext(self, self._ctx, self.state)
        self.enterRule(localctx, 256, self.RULE_errorCapturingIdentifierExtra)
        try:
            self.state = 2923
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 376, self._ctx)
            if la_ == 1:
                localctx = sqlParser.ErrorIdentContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2918
                self._errHandler.sync(self)
                _alt = 1
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2916
                        self.match(sqlParser.MINUS)
                        self.state = 2917
                        self.identifier()
                    else:
                        raise NoViableAltException(self)
                    self.state = 2920
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 375, self._ctx)
                pass
            elif la_ == 2:
                localctx = sqlParser.RealIdentContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class IdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def strictIdentifier(self):
            return self.getTypedRuleContext(sqlParser.StrictIdentifierContext, 0)

        def strictNonReserved(self):
            return self.getTypedRuleContext(sqlParser.StrictNonReservedContext, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_identifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifier'):
                return visitor.visitIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def identifier(self):
        localctx = sqlParser.IdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 258, self.RULE_identifier)
        try:
            self.state = 2928
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 377, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2925
                self.strictIdentifier()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2926
                if not not self.SQL_standard_keyword_behavior:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'not self.SQL_standard_keyword_behavior')
                self.state = 2927
                self.strictNonReserved()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StrictIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_strictIdentifier

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class QuotedIdentifierAlternativeContext(StrictIdentifierContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def quotedIdentifier(self):
            return self.getTypedRuleContext(sqlParser.QuotedIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQuotedIdentifierAlternative'):
                return visitor.visitQuotedIdentifierAlternative(self)
            else:
                return visitor.visitChildren(self)

    class UnquotedIdentifierContext(StrictIdentifierContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def IDENTIFIER(self):
            return self.getToken(sqlParser.IDENTIFIER, 0)

        def ansiNonReserved(self):
            return self.getTypedRuleContext(sqlParser.AnsiNonReservedContext, 0)

        def nonReserved(self):
            return self.getTypedRuleContext(sqlParser.NonReservedContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnquotedIdentifier'):
                return visitor.visitUnquotedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def strictIdentifier(self):
        localctx = sqlParser.StrictIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 260, self.RULE_strictIdentifier)
        try:
            self.state = 2936
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 378, self._ctx)
            if la_ == 1:
                localctx = sqlParser.UnquotedIdentifierContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2930
                self.match(sqlParser.IDENTIFIER)
                pass
            elif la_ == 2:
                localctx = sqlParser.QuotedIdentifierAlternativeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2931
                self.quotedIdentifier()
                pass
            elif la_ == 3:
                localctx = sqlParser.UnquotedIdentifierContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2932
                if not self.SQL_standard_keyword_behavior:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'self.SQL_standard_keyword_behavior')
                self.state = 2933
                self.ansiNonReserved()
                pass
            elif la_ == 4:
                localctx = sqlParser.UnquotedIdentifierContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2934
                if not not self.SQL_standard_keyword_behavior:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'not self.SQL_standard_keyword_behavior')
                self.state = 2935
                self.nonReserved()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class QuotedIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BACKQUOTED_IDENTIFIER(self):
            return self.getToken(sqlParser.BACKQUOTED_IDENTIFIER, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_quotedIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQuotedIdentifier'):
                return visitor.visitQuotedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def quotedIdentifier(self):
        localctx = sqlParser.QuotedIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 262, self.RULE_quotedIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2938
            self.match(sqlParser.BACKQUOTED_IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NumberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return sqlParser.RULE_number

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class DecimalLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DECIMAL_VALUE(self):
            return self.getToken(sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDecimalLiteral'):
                return visitor.visitDecimalLiteral(self)
            else:
                return visitor.visitChildren(self)

    class BigIntLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def BIGINT_LITERAL(self):
            return self.getToken(sqlParser.BIGINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBigIntLiteral'):
                return visitor.visitBigIntLiteral(self)
            else:
                return visitor.visitChildren(self)

    class TinyIntLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def TINYINT_LITERAL(self):
            return self.getToken(sqlParser.TINYINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTinyIntLiteral'):
                return visitor.visitTinyIntLiteral(self)
            else:
                return visitor.visitChildren(self)

    class LegacyDecimalLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXPONENT_VALUE(self):
            return self.getToken(sqlParser.EXPONENT_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLegacyDecimalLiteral'):
                return visitor.visitLegacyDecimalLiteral(self)
            else:
                return visitor.visitChildren(self)

    class BigDecimalLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def BIGDECIMAL_LITERAL(self):
            return self.getToken(sqlParser.BIGDECIMAL_LITERAL, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBigDecimalLiteral'):
                return visitor.visitBigDecimalLiteral(self)
            else:
                return visitor.visitChildren(self)

    class ExponentLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXPONENT_VALUE(self):
            return self.getToken(sqlParser.EXPONENT_VALUE, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExponentLiteral'):
                return visitor.visitExponentLiteral(self)
            else:
                return visitor.visitChildren(self)

    class DoubleLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DOUBLE_LITERAL(self):
            return self.getToken(sqlParser.DOUBLE_LITERAL, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDoubleLiteral'):
                return visitor.visitDoubleLiteral(self)
            else:
                return visitor.visitChildren(self)

    class IntegerLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def INTEGER_VALUE(self):
            return self.getToken(sqlParser.INTEGER_VALUE, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntegerLiteral'):
                return visitor.visitIntegerLiteral(self)
            else:
                return visitor.visitChildren(self)

    class SmallIntLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def SMALLINT_LITERAL(self):
            return self.getToken(sqlParser.SMALLINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSmallIntLiteral'):
                return visitor.visitSmallIntLiteral(self)
            else:
                return visitor.visitChildren(self)

    def number(self):
        localctx = sqlParser.NumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 264, self.RULE_number)
        self._la = 0
        try:
            self.state = 2979
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 388, self._ctx)
            if la_ == 1:
                localctx = sqlParser.ExponentLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2940
                if not not self.legacy_exponent_literal_as_decimal_enabled:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'not self.legacy_exponent_literal_as_decimal_enabled')
                self.state = 2942
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2941
                    self.match(sqlParser.MINUS)
                self.state = 2944
                self.match(sqlParser.EXPONENT_VALUE)
                pass
            elif la_ == 2:
                localctx = sqlParser.DecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2945
                if not not self.legacy_exponent_literal_as_decimal_enabled:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'not self.legacy_exponent_literal_as_decimal_enabled')
                self.state = 2947
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2946
                    self.match(sqlParser.MINUS)
                self.state = 2949
                self.match(sqlParser.DECIMAL_VALUE)
                pass
            elif la_ == 3:
                localctx = sqlParser.LegacyDecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2950
                if not self.legacy_exponent_literal_as_decimal_enabled:
                    from antlr4.error.Errors import FailedPredicateException
                    raise FailedPredicateException(self, 'self.legacy_exponent_literal_as_decimal_enabled')
                self.state = 2952
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2951
                    self.match(sqlParser.MINUS)
                self.state = 2954
                _la = self._input.LA(1)
                if not (_la == 288 or _la == 289):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 4:
                localctx = sqlParser.IntegerLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2956
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2955
                    self.match(sqlParser.MINUS)
                self.state = 2958
                self.match(sqlParser.INTEGER_VALUE)
                pass
            elif la_ == 5:
                localctx = sqlParser.BigIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 2960
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2959
                    self.match(sqlParser.MINUS)
                self.state = 2962
                self.match(sqlParser.BIGINT_LITERAL)
                pass
            elif la_ == 6:
                localctx = sqlParser.SmallIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 2964
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2963
                    self.match(sqlParser.MINUS)
                self.state = 2966
                self.match(sqlParser.SMALLINT_LITERAL)
                pass
            elif la_ == 7:
                localctx = sqlParser.TinyIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 2968
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2967
                    self.match(sqlParser.MINUS)
                self.state = 2970
                self.match(sqlParser.TINYINT_LITERAL)
                pass
            elif la_ == 8:
                localctx = sqlParser.DoubleLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 2972
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2971
                    self.match(sqlParser.MINUS)
                self.state = 2974
                self.match(sqlParser.DOUBLE_LITERAL)
                pass
            elif la_ == 9:
                localctx = sqlParser.BigDecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 2976
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 273:
                    self.state = 2975
                    self.match(sqlParser.MINUS)
                self.state = 2978
                self.match(sqlParser.BIGDECIMAL_LITERAL)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AlterColumnActionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.setOrDrop = None

        def TYPE(self):
            return self.getToken(sqlParser.TYPE, 0)

        def dataType(self):
            return self.getTypedRuleContext(sqlParser.DataTypeContext, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(sqlParser.CommentSpecContext, 0)

        def colPosition(self):
            return self.getTypedRuleContext(sqlParser.ColPositionContext, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_alterColumnAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAlterColumnAction'):
                return visitor.visitAlterColumnAction(self)
            else:
                return visitor.visitChildren(self)

    def alterColumnAction(self):
        localctx = sqlParser.AlterColumnActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 266, self.RULE_alterColumnAction)
        self._la = 0
        try:
            self.state = 2988
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [243]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2981
                self.match(sqlParser.TYPE)
                self.state = 2982
                self.dataType()
                pass
            elif token in [45]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2983
                self.commentSpec()
                pass
            elif token in [13, 94]:
                self.enterOuterAlt(localctx, 3)
                self.state = 2984
                self.colPosition()
                pass
            elif token in [76, 212]:
                self.enterOuterAlt(localctx, 4)
                self.state = 2985
                localctx.setOrDrop = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 76 or _la == 212):
                    localctx.setOrDrop = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2986
                self.match(sqlParser.NOT)
                self.state = 2987
                self.match(sqlParser.NULL)
                pass
            else:
                raise NoViableAltException(self)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AnsiNonReservedContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ADD(self):
            return self.getToken(sqlParser.ADD, 0)

        def AFTER(self):
            return self.getToken(sqlParser.AFTER, 0)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def ANALYZE(self):
            return self.getToken(sqlParser.ANALYZE, 0)

        def ARCHIVE(self):
            return self.getToken(sqlParser.ARCHIVE, 0)

        def ARRAY(self):
            return self.getToken(sqlParser.ARRAY, 0)

        def ASC(self):
            return self.getToken(sqlParser.ASC, 0)

        def AT(self):
            return self.getToken(sqlParser.AT, 0)

        def BETWEEN(self):
            return self.getToken(sqlParser.BETWEEN, 0)

        def BUCKET(self):
            return self.getToken(sqlParser.BUCKET, 0)

        def BUCKETS(self):
            return self.getToken(sqlParser.BUCKETS, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def CACHE(self):
            return self.getToken(sqlParser.CACHE, 0)

        def CASCADE(self):
            return self.getToken(sqlParser.CASCADE, 0)

        def CHANGE(self):
            return self.getToken(sqlParser.CHANGE, 0)

        def CLEAR(self):
            return self.getToken(sqlParser.CLEAR, 0)

        def CLUSTER(self):
            return self.getToken(sqlParser.CLUSTER, 0)

        def CLUSTERED(self):
            return self.getToken(sqlParser.CLUSTERED, 0)

        def CODEGEN(self):
            return self.getToken(sqlParser.CODEGEN, 0)

        def COLLECTION(self):
            return self.getToken(sqlParser.COLLECTION, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def COMMENT(self):
            return self.getToken(sqlParser.COMMENT, 0)

        def COMMIT(self):
            return self.getToken(sqlParser.COMMIT, 0)

        def COMPACT(self):
            return self.getToken(sqlParser.COMPACT, 0)

        def COMPACTIONS(self):
            return self.getToken(sqlParser.COMPACTIONS, 0)

        def COMPUTE(self):
            return self.getToken(sqlParser.COMPUTE, 0)

        def CONCATENATE(self):
            return self.getToken(sqlParser.CONCATENATE, 0)

        def COST(self):
            return self.getToken(sqlParser.COST, 0)

        def CUBE(self):
            return self.getToken(sqlParser.CUBE, 0)

        def CURRENT(self):
            return self.getToken(sqlParser.CURRENT, 0)

        def DATA(self):
            return self.getToken(sqlParser.DATA, 0)

        def DATABASE(self):
            return self.getToken(sqlParser.DATABASE, 0)

        def DATABASES(self):
            return self.getToken(sqlParser.DATABASES, 0)

        def DBPROPERTIES(self):
            return self.getToken(sqlParser.DBPROPERTIES, 0)

        def DEFINED(self):
            return self.getToken(sqlParser.DEFINED, 0)

        def DELETE(self):
            return self.getToken(sqlParser.DELETE, 0)

        def DELIMITED(self):
            return self.getToken(sqlParser.DELIMITED, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def DFS(self):
            return self.getToken(sqlParser.DFS, 0)

        def DIRECTORIES(self):
            return self.getToken(sqlParser.DIRECTORIES, 0)

        def DIRECTORY(self):
            return self.getToken(sqlParser.DIRECTORY, 0)

        def DISTRIBUTE(self):
            return self.getToken(sqlParser.DISTRIBUTE, 0)

        def DIV(self):
            return self.getToken(sqlParser.DIV, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def ESCAPED(self):
            return self.getToken(sqlParser.ESCAPED, 0)

        def EXCHANGE(self):
            return self.getToken(sqlParser.EXCHANGE, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def EXPLAIN(self):
            return self.getToken(sqlParser.EXPLAIN, 0)

        def EXPORT(self):
            return self.getToken(sqlParser.EXPORT, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def EXTERNAL(self):
            return self.getToken(sqlParser.EXTERNAL, 0)

        def EXTRACT(self):
            return self.getToken(sqlParser.EXTRACT, 0)

        def FIELDS(self):
            return self.getToken(sqlParser.FIELDS, 0)

        def FILEFORMAT(self):
            return self.getToken(sqlParser.FILEFORMAT, 0)

        def FIRST(self):
            return self.getToken(sqlParser.FIRST, 0)

        def FOLLOWING(self):
            return self.getToken(sqlParser.FOLLOWING, 0)

        def FORMAT(self):
            return self.getToken(sqlParser.FORMAT, 0)

        def FORMATTED(self):
            return self.getToken(sqlParser.FORMATTED, 0)

        def FUNCTION(self):
            return self.getToken(sqlParser.FUNCTION, 0)

        def FUNCTIONS(self):
            return self.getToken(sqlParser.FUNCTIONS, 0)

        def GLOBAL(self):
            return self.getToken(sqlParser.GLOBAL, 0)

        def GROUPING(self):
            return self.getToken(sqlParser.GROUPING, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def IGNORE(self):
            return self.getToken(sqlParser.IGNORE, 0)

        def IMPORT(self):
            return self.getToken(sqlParser.IMPORT, 0)

        def INDEX(self):
            return self.getToken(sqlParser.INDEX, 0)

        def INDEXES(self):
            return self.getToken(sqlParser.INDEXES, 0)

        def INPATH(self):
            return self.getToken(sqlParser.INPATH, 0)

        def INPUTFORMAT(self):
            return self.getToken(sqlParser.INPUTFORMAT, 0)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def INTERVAL(self):
            return self.getToken(sqlParser.INTERVAL, 0)

        def ITEMS(self):
            return self.getToken(sqlParser.ITEMS, 0)

        def KEYS(self):
            return self.getToken(sqlParser.KEYS, 0)

        def LAST(self):
            return self.getToken(sqlParser.LAST, 0)

        def LATERAL(self):
            return self.getToken(sqlParser.LATERAL, 0)

        def LAZY(self):
            return self.getToken(sqlParser.LAZY, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def LIMIT(self):
            return self.getToken(sqlParser.LIMIT, 0)

        def LINES(self):
            return self.getToken(sqlParser.LINES, 0)

        def LIST(self):
            return self.getToken(sqlParser.LIST, 0)

        def LOAD(self):
            return self.getToken(sqlParser.LOAD, 0)

        def LOCAL(self):
            return self.getToken(sqlParser.LOCAL, 0)

        def LOCATION(self):
            return self.getToken(sqlParser.LOCATION, 0)

        def LOCK(self):
            return self.getToken(sqlParser.LOCK, 0)

        def LOCKS(self):
            return self.getToken(sqlParser.LOCKS, 0)

        def LOGICAL(self):
            return self.getToken(sqlParser.LOGICAL, 0)

        def MACRO(self):
            return self.getToken(sqlParser.MACRO, 0)

        def MAP(self):
            return self.getToken(sqlParser.MAP, 0)

        def MATCHED(self):
            return self.getToken(sqlParser.MATCHED, 0)

        def MERGE(self):
            return self.getToken(sqlParser.MERGE, 0)

        def MSCK(self):
            return self.getToken(sqlParser.MSCK, 0)

        def NAMESPACE(self):
            return self.getToken(sqlParser.NAMESPACE, 0)

        def NAMESPACES(self):
            return self.getToken(sqlParser.NAMESPACES, 0)

        def NO(self):
            return self.getToken(sqlParser.NO, 0)

        def NULLS(self):
            return self.getToken(sqlParser.NULLS, 0)

        def OF(self):
            return self.getToken(sqlParser.OF, 0)

        def OPTION(self):
            return self.getToken(sqlParser.OPTION, 0)

        def OPTIONS(self):
            return self.getToken(sqlParser.OPTIONS, 0)

        def OUT(self):
            return self.getToken(sqlParser.OUT, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(sqlParser.OUTPUTFORMAT, 0)

        def OVER(self):
            return self.getToken(sqlParser.OVER, 0)

        def OVERLAY(self):
            return self.getToken(sqlParser.OVERLAY, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def PARTITION(self):
            return self.getToken(sqlParser.PARTITION, 0)

        def PARTITIONED(self):
            return self.getToken(sqlParser.PARTITIONED, 0)

        def PARTITIONS(self):
            return self.getToken(sqlParser.PARTITIONS, 0)

        def PERCENTLIT(self):
            return self.getToken(sqlParser.PERCENTLIT, 0)

        def PIVOT(self):
            return self.getToken(sqlParser.PIVOT, 0)

        def PLACING(self):
            return self.getToken(sqlParser.PLACING, 0)

        def POSITION(self):
            return self.getToken(sqlParser.POSITION, 0)

        def PRECEDING(self):
            return self.getToken(sqlParser.PRECEDING, 0)

        def PRINCIPALS(self):
            return self.getToken(sqlParser.PRINCIPALS, 0)

        def PROPERTIES(self):
            return self.getToken(sqlParser.PROPERTIES, 0)

        def PURGE(self):
            return self.getToken(sqlParser.PURGE, 0)

        def QUERY(self):
            return self.getToken(sqlParser.QUERY, 0)

        def RANGE(self):
            return self.getToken(sqlParser.RANGE, 0)

        def RECORDREADER(self):
            return self.getToken(sqlParser.RECORDREADER, 0)

        def RECORDWRITER(self):
            return self.getToken(sqlParser.RECORDWRITER, 0)

        def RECOVER(self):
            return self.getToken(sqlParser.RECOVER, 0)

        def REDUCE(self):
            return self.getToken(sqlParser.REDUCE, 0)

        def REFRESH(self):
            return self.getToken(sqlParser.REFRESH, 0)

        def RENAME(self):
            return self.getToken(sqlParser.RENAME, 0)

        def REPAIR(self):
            return self.getToken(sqlParser.REPAIR, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def RESET(self):
            return self.getToken(sqlParser.RESET, 0)

        def RESTRICT(self):
            return self.getToken(sqlParser.RESTRICT, 0)

        def REVOKE(self):
            return self.getToken(sqlParser.REVOKE, 0)

        def RLIKE(self):
            return self.getToken(sqlParser.RLIKE, 0)

        def ROLE(self):
            return self.getToken(sqlParser.ROLE, 0)

        def ROLES(self):
            return self.getToken(sqlParser.ROLES, 0)

        def ROLLBACK(self):
            return self.getToken(sqlParser.ROLLBACK, 0)

        def ROLLUP(self):
            return self.getToken(sqlParser.ROLLUP, 0)

        def ROW(self):
            return self.getToken(sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(sqlParser.ROWS, 0)

        def SCHEMA(self):
            return self.getToken(sqlParser.SCHEMA, 0)

        def SEPARATED(self):
            return self.getToken(sqlParser.SEPARATED, 0)

        def SERDE(self):
            return self.getToken(sqlParser.SERDE, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(sqlParser.SERDEPROPERTIES, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def SETS(self):
            return self.getToken(sqlParser.SETS, 0)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def SKEWED(self):
            return self.getToken(sqlParser.SKEWED, 0)

        def SORT(self):
            return self.getToken(sqlParser.SORT, 0)

        def SORTED(self):
            return self.getToken(sqlParser.SORTED, 0)

        def START(self):
            return self.getToken(sqlParser.START, 0)

        def STATISTICS(self):
            return self.getToken(sqlParser.STATISTICS, 0)

        def STORED(self):
            return self.getToken(sqlParser.STORED, 0)

        def STRATIFY(self):
            return self.getToken(sqlParser.STRATIFY, 0)

        def STRUCT(self):
            return self.getToken(sqlParser.STRUCT, 0)

        def SUBSTR(self):
            return self.getToken(sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(sqlParser.SUBSTRING, 0)

        def TABLES(self):
            return self.getToken(sqlParser.TABLES, 0)

        def TABLESAMPLE(self):
            return self.getToken(sqlParser.TABLESAMPLE, 0)

        def TBLPROPERTIES(self):
            return self.getToken(sqlParser.TBLPROPERTIES, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def TERMINATED(self):
            return self.getToken(sqlParser.TERMINATED, 0)

        def TOUCH(self):
            return self.getToken(sqlParser.TOUCH, 0)

        def TRANSACTION(self):
            return self.getToken(sqlParser.TRANSACTION, 0)

        def TRANSACTIONS(self):
            return self.getToken(sqlParser.TRANSACTIONS, 0)

        def TRANSFORM(self):
            return self.getToken(sqlParser.TRANSFORM, 0)

        def TRIM(self):
            return self.getToken(sqlParser.TRIM, 0)

        def TRUE(self):
            return self.getToken(sqlParser.TRUE, 0)

        def TRUNCATE(self):
            return self.getToken(sqlParser.TRUNCATE, 0)

        def UNARCHIVE(self):
            return self.getToken(sqlParser.UNARCHIVE, 0)

        def UNBOUNDED(self):
            return self.getToken(sqlParser.UNBOUNDED, 0)

        def UNCACHE(self):
            return self.getToken(sqlParser.UNCACHE, 0)

        def UNLOCK(self):
            return self.getToken(sqlParser.UNLOCK, 0)

        def UNSET(self):
            return self.getToken(sqlParser.UNSET, 0)

        def UPDATE(self):
            return self.getToken(sqlParser.UPDATE, 0)

        def USE(self):
            return self.getToken(sqlParser.USE, 0)

        def VALUES(self):
            return self.getToken(sqlParser.VALUES, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def VIEWS(self):
            return self.getToken(sqlParser.VIEWS, 0)

        def WINDOW(self):
            return self.getToken(sqlParser.WINDOW, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_ansiNonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAnsiNonReserved'):
                return visitor.visitAnsiNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def ansiNonReserved(self):
        localctx = sqlParser.AnsiNonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 268, self.RULE_ansiNonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2990
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & -2191012308494209024 != 0 or (_la - 65 & ~63 == 0 and 1 << _la - 65 & -1623858937164034561 != 0) or (_la - 129 & ~63 == 0 and 1 << _la - 129 & -577024025239617543 != 0) or (_la - 193 & ~63 == 0 and 1 << _la - 193 & -7044767828940189705 != 0) or (_la - 257 & ~63 == 0 and 1 << _la - 257 & 1048595 != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StrictNonReservedContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ANTI(self):
            return self.getToken(sqlParser.ANTI, 0)

        def CROSS(self):
            return self.getToken(sqlParser.CROSS, 0)

        def EXCEPT(self):
            return self.getToken(sqlParser.EXCEPT, 0)

        def FULL(self):
            return self.getToken(sqlParser.FULL, 0)

        def INNER(self):
            return self.getToken(sqlParser.INNER, 0)

        def INTERSECT(self):
            return self.getToken(sqlParser.INTERSECT, 0)

        def JOIN(self):
            return self.getToken(sqlParser.JOIN, 0)

        def LEFT(self):
            return self.getToken(sqlParser.LEFT, 0)

        def NATURAL(self):
            return self.getToken(sqlParser.NATURAL, 0)

        def ON(self):
            return self.getToken(sqlParser.ON, 0)

        def RIGHT(self):
            return self.getToken(sqlParser.RIGHT, 0)

        def SEMI(self):
            return self.getToken(sqlParser.SEMI, 0)

        def SETMINUS(self):
            return self.getToken(sqlParser.SETMINUS, 0)

        def UNION(self):
            return self.getToken(sqlParser.UNION, 0)

        def USING(self):
            return self.getToken(sqlParser.USING, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_strictNonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStrictNonReserved'):
                return visitor.visitStrictNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def strictNonReserved(self):
        localctx = sqlParser.StrictNonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 270, self.RULE_strictNonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2992
            _la = self._input.LA(1)
            if not (_la - 18 & ~63 == 0 and 1 << _la - 18 & -9223371968135299071 != 0 or (_la - 101 & ~63 == 0 and 1 << _la - 101 & 73183495035846657 != 0) or (_la - 196 & ~63 == 0 and 1 << _la - 196 & 578712552117241857 != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class NonReservedContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ADD(self):
            return self.getToken(sqlParser.ADD, 0)

        def AFTER(self):
            return self.getToken(sqlParser.AFTER, 0)

        def ALL(self):
            return self.getToken(sqlParser.ALL, 0)

        def ALTER(self):
            return self.getToken(sqlParser.ALTER, 0)

        def ANALYZE(self):
            return self.getToken(sqlParser.ANALYZE, 0)

        def AND(self):
            return self.getToken(sqlParser.AND, 0)

        def ANY(self):
            return self.getToken(sqlParser.ANY, 0)

        def ARCHIVE(self):
            return self.getToken(sqlParser.ARCHIVE, 0)

        def ARRAY(self):
            return self.getToken(sqlParser.ARRAY, 0)

        def AS(self):
            return self.getToken(sqlParser.AS, 0)

        def ASC(self):
            return self.getToken(sqlParser.ASC, 0)

        def AT(self):
            return self.getToken(sqlParser.AT, 0)

        def AUTHORIZATION(self):
            return self.getToken(sqlParser.AUTHORIZATION, 0)

        def BETWEEN(self):
            return self.getToken(sqlParser.BETWEEN, 0)

        def BOTH(self):
            return self.getToken(sqlParser.BOTH, 0)

        def BUCKET(self):
            return self.getToken(sqlParser.BUCKET, 0)

        def BUCKETS(self):
            return self.getToken(sqlParser.BUCKETS, 0)

        def BY(self):
            return self.getToken(sqlParser.BY, 0)

        def CACHE(self):
            return self.getToken(sqlParser.CACHE, 0)

        def CASCADE(self):
            return self.getToken(sqlParser.CASCADE, 0)

        def CASE(self):
            return self.getToken(sqlParser.CASE, 0)

        def CAST(self):
            return self.getToken(sqlParser.CAST, 0)

        def CHANGE(self):
            return self.getToken(sqlParser.CHANGE, 0)

        def CHECK(self):
            return self.getToken(sqlParser.CHECK, 0)

        def CLEAR(self):
            return self.getToken(sqlParser.CLEAR, 0)

        def CLUSTER(self):
            return self.getToken(sqlParser.CLUSTER, 0)

        def CLUSTERED(self):
            return self.getToken(sqlParser.CLUSTERED, 0)

        def CODEGEN(self):
            return self.getToken(sqlParser.CODEGEN, 0)

        def COLLATE(self):
            return self.getToken(sqlParser.COLLATE, 0)

        def COLLECTION(self):
            return self.getToken(sqlParser.COLLECTION, 0)

        def COLUMN(self):
            return self.getToken(sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(sqlParser.COLUMNS, 0)

        def COMMENT(self):
            return self.getToken(sqlParser.COMMENT, 0)

        def COMMIT(self):
            return self.getToken(sqlParser.COMMIT, 0)

        def COMPACT(self):
            return self.getToken(sqlParser.COMPACT, 0)

        def COMPACTIONS(self):
            return self.getToken(sqlParser.COMPACTIONS, 0)

        def COMPUTE(self):
            return self.getToken(sqlParser.COMPUTE, 0)

        def CONCATENATE(self):
            return self.getToken(sqlParser.CONCATENATE, 0)

        def CONSTRAINT(self):
            return self.getToken(sqlParser.CONSTRAINT, 0)

        def COST(self):
            return self.getToken(sqlParser.COST, 0)

        def CREATE(self):
            return self.getToken(sqlParser.CREATE, 0)

        def CUBE(self):
            return self.getToken(sqlParser.CUBE, 0)

        def CURRENT(self):
            return self.getToken(sqlParser.CURRENT, 0)

        def CURRENT_DATE(self):
            return self.getToken(sqlParser.CURRENT_DATE, 0)

        def CURRENT_TIME(self):
            return self.getToken(sqlParser.CURRENT_TIME, 0)

        def CURRENT_TIMESTAMP(self):
            return self.getToken(sqlParser.CURRENT_TIMESTAMP, 0)

        def CURRENT_USER(self):
            return self.getToken(sqlParser.CURRENT_USER, 0)

        def DATA(self):
            return self.getToken(sqlParser.DATA, 0)

        def DATABASE(self):
            return self.getToken(sqlParser.DATABASE, 0)

        def DATABASES(self):
            return self.getToken(sqlParser.DATABASES, 0)

        def DAY(self):
            return self.getToken(sqlParser.DAY, 0)

        def DBPROPERTIES(self):
            return self.getToken(sqlParser.DBPROPERTIES, 0)

        def DEFINED(self):
            return self.getToken(sqlParser.DEFINED, 0)

        def DELETE(self):
            return self.getToken(sqlParser.DELETE, 0)

        def DELIMITED(self):
            return self.getToken(sqlParser.DELIMITED, 0)

        def DESC(self):
            return self.getToken(sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(sqlParser.DESCRIBE, 0)

        def DFS(self):
            return self.getToken(sqlParser.DFS, 0)

        def DIRECTORIES(self):
            return self.getToken(sqlParser.DIRECTORIES, 0)

        def DIRECTORY(self):
            return self.getToken(sqlParser.DIRECTORY, 0)

        def DISTINCT(self):
            return self.getToken(sqlParser.DISTINCT, 0)

        def DISTRIBUTE(self):
            return self.getToken(sqlParser.DISTRIBUTE, 0)

        def DIV(self):
            return self.getToken(sqlParser.DIV, 0)

        def DROP(self):
            return self.getToken(sqlParser.DROP, 0)

        def ELSE(self):
            return self.getToken(sqlParser.ELSE, 0)

        def END(self):
            return self.getToken(sqlParser.END, 0)

        def ESCAPE(self):
            return self.getToken(sqlParser.ESCAPE, 0)

        def ESCAPED(self):
            return self.getToken(sqlParser.ESCAPED, 0)

        def EXCHANGE(self):
            return self.getToken(sqlParser.EXCHANGE, 0)

        def EXISTS(self):
            return self.getToken(sqlParser.EXISTS, 0)

        def EXPLAIN(self):
            return self.getToken(sqlParser.EXPLAIN, 0)

        def EXPORT(self):
            return self.getToken(sqlParser.EXPORT, 0)

        def EXTENDED(self):
            return self.getToken(sqlParser.EXTENDED, 0)

        def EXTERNAL(self):
            return self.getToken(sqlParser.EXTERNAL, 0)

        def EXTRACT(self):
            return self.getToken(sqlParser.EXTRACT, 0)

        def FALSE(self):
            return self.getToken(sqlParser.FALSE, 0)

        def FETCH(self):
            return self.getToken(sqlParser.FETCH, 0)

        def FILTER(self):
            return self.getToken(sqlParser.FILTER, 0)

        def FIELDS(self):
            return self.getToken(sqlParser.FIELDS, 0)

        def FILEFORMAT(self):
            return self.getToken(sqlParser.FILEFORMAT, 0)

        def FIRST(self):
            return self.getToken(sqlParser.FIRST, 0)

        def FOLLOWING(self):
            return self.getToken(sqlParser.FOLLOWING, 0)

        def FOR(self):
            return self.getToken(sqlParser.FOR, 0)

        def FOREIGN(self):
            return self.getToken(sqlParser.FOREIGN, 0)

        def FORMAT(self):
            return self.getToken(sqlParser.FORMAT, 0)

        def FORMATTED(self):
            return self.getToken(sqlParser.FORMATTED, 0)

        def FROM(self):
            return self.getToken(sqlParser.FROM, 0)

        def FUNCTION(self):
            return self.getToken(sqlParser.FUNCTION, 0)

        def FUNCTIONS(self):
            return self.getToken(sqlParser.FUNCTIONS, 0)

        def GLOBAL(self):
            return self.getToken(sqlParser.GLOBAL, 0)

        def GRANT(self):
            return self.getToken(sqlParser.GRANT, 0)

        def GROUP(self):
            return self.getToken(sqlParser.GROUP, 0)

        def GROUPING(self):
            return self.getToken(sqlParser.GROUPING, 0)

        def HAVING(self):
            return self.getToken(sqlParser.HAVING, 0)

        def HOUR(self):
            return self.getToken(sqlParser.HOUR, 0)

        def IF(self):
            return self.getToken(sqlParser.IF, 0)

        def IGNORE(self):
            return self.getToken(sqlParser.IGNORE, 0)

        def IMPORT(self):
            return self.getToken(sqlParser.IMPORT, 0)

        def IN(self):
            return self.getToken(sqlParser.IN, 0)

        def INDEX(self):
            return self.getToken(sqlParser.INDEX, 0)

        def INDEXES(self):
            return self.getToken(sqlParser.INDEXES, 0)

        def INPATH(self):
            return self.getToken(sqlParser.INPATH, 0)

        def INPUTFORMAT(self):
            return self.getToken(sqlParser.INPUTFORMAT, 0)

        def INSERT(self):
            return self.getToken(sqlParser.INSERT, 0)

        def INTERVAL(self):
            return self.getToken(sqlParser.INTERVAL, 0)

        def INTO(self):
            return self.getToken(sqlParser.INTO, 0)

        def IS(self):
            return self.getToken(sqlParser.IS, 0)

        def ITEMS(self):
            return self.getToken(sqlParser.ITEMS, 0)

        def KEYS(self):
            return self.getToken(sqlParser.KEYS, 0)

        def LAST(self):
            return self.getToken(sqlParser.LAST, 0)

        def LATERAL(self):
            return self.getToken(sqlParser.LATERAL, 0)

        def LAZY(self):
            return self.getToken(sqlParser.LAZY, 0)

        def LEADING(self):
            return self.getToken(sqlParser.LEADING, 0)

        def LIKE(self):
            return self.getToken(sqlParser.LIKE, 0)

        def LIMIT(self):
            return self.getToken(sqlParser.LIMIT, 0)

        def LINES(self):
            return self.getToken(sqlParser.LINES, 0)

        def LIST(self):
            return self.getToken(sqlParser.LIST, 0)

        def LOAD(self):
            return self.getToken(sqlParser.LOAD, 0)

        def LOCAL(self):
            return self.getToken(sqlParser.LOCAL, 0)

        def LOCATION(self):
            return self.getToken(sqlParser.LOCATION, 0)

        def LOCK(self):
            return self.getToken(sqlParser.LOCK, 0)

        def LOCKS(self):
            return self.getToken(sqlParser.LOCKS, 0)

        def LOGICAL(self):
            return self.getToken(sqlParser.LOGICAL, 0)

        def MACRO(self):
            return self.getToken(sqlParser.MACRO, 0)

        def MAP(self):
            return self.getToken(sqlParser.MAP, 0)

        def MATCHED(self):
            return self.getToken(sqlParser.MATCHED, 0)

        def MERGE(self):
            return self.getToken(sqlParser.MERGE, 0)

        def MINUTE(self):
            return self.getToken(sqlParser.MINUTE, 0)

        def MONTH(self):
            return self.getToken(sqlParser.MONTH, 0)

        def MSCK(self):
            return self.getToken(sqlParser.MSCK, 0)

        def NAMESPACE(self):
            return self.getToken(sqlParser.NAMESPACE, 0)

        def NAMESPACES(self):
            return self.getToken(sqlParser.NAMESPACES, 0)

        def NO(self):
            return self.getToken(sqlParser.NO, 0)

        def NOT(self):
            return self.getToken(sqlParser.NOT, 0)

        def NULL(self):
            return self.getToken(sqlParser.NULL, 0)

        def NULLS(self):
            return self.getToken(sqlParser.NULLS, 0)

        def OF(self):
            return self.getToken(sqlParser.OF, 0)

        def ONLY(self):
            return self.getToken(sqlParser.ONLY, 0)

        def OPTION(self):
            return self.getToken(sqlParser.OPTION, 0)

        def OPTIONS(self):
            return self.getToken(sqlParser.OPTIONS, 0)

        def OR(self):
            return self.getToken(sqlParser.OR, 0)

        def ORDER(self):
            return self.getToken(sqlParser.ORDER, 0)

        def OUT(self):
            return self.getToken(sqlParser.OUT, 0)

        def OUTER(self):
            return self.getToken(sqlParser.OUTER, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(sqlParser.OUTPUTFORMAT, 0)

        def OVER(self):
            return self.getToken(sqlParser.OVER, 0)

        def OVERLAPS(self):
            return self.getToken(sqlParser.OVERLAPS, 0)

        def OVERLAY(self):
            return self.getToken(sqlParser.OVERLAY, 0)

        def OVERWRITE(self):
            return self.getToken(sqlParser.OVERWRITE, 0)

        def PARTITION(self):
            return self.getToken(sqlParser.PARTITION, 0)

        def PARTITIONED(self):
            return self.getToken(sqlParser.PARTITIONED, 0)

        def PARTITIONS(self):
            return self.getToken(sqlParser.PARTITIONS, 0)

        def PERCENTLIT(self):
            return self.getToken(sqlParser.PERCENTLIT, 0)

        def PIVOT(self):
            return self.getToken(sqlParser.PIVOT, 0)

        def PLACING(self):
            return self.getToken(sqlParser.PLACING, 0)

        def POSITION(self):
            return self.getToken(sqlParser.POSITION, 0)

        def PRECEDING(self):
            return self.getToken(sqlParser.PRECEDING, 0)

        def PRIMARY(self):
            return self.getToken(sqlParser.PRIMARY, 0)

        def PRINCIPALS(self):
            return self.getToken(sqlParser.PRINCIPALS, 0)

        def PROPERTIES(self):
            return self.getToken(sqlParser.PROPERTIES, 0)

        def PURGE(self):
            return self.getToken(sqlParser.PURGE, 0)

        def QUERY(self):
            return self.getToken(sqlParser.QUERY, 0)

        def RANGE(self):
            return self.getToken(sqlParser.RANGE, 0)

        def RECORDREADER(self):
            return self.getToken(sqlParser.RECORDREADER, 0)

        def RECORDWRITER(self):
            return self.getToken(sqlParser.RECORDWRITER, 0)

        def RECOVER(self):
            return self.getToken(sqlParser.RECOVER, 0)

        def REDUCE(self):
            return self.getToken(sqlParser.REDUCE, 0)

        def REFERENCES(self):
            return self.getToken(sqlParser.REFERENCES, 0)

        def REFRESH(self):
            return self.getToken(sqlParser.REFRESH, 0)

        def RENAME(self):
            return self.getToken(sqlParser.RENAME, 0)

        def REPAIR(self):
            return self.getToken(sqlParser.REPAIR, 0)

        def REPLACE(self):
            return self.getToken(sqlParser.REPLACE, 0)

        def RESET(self):
            return self.getToken(sqlParser.RESET, 0)

        def RESTRICT(self):
            return self.getToken(sqlParser.RESTRICT, 0)

        def REVOKE(self):
            return self.getToken(sqlParser.REVOKE, 0)

        def RLIKE(self):
            return self.getToken(sqlParser.RLIKE, 0)

        def ROLE(self):
            return self.getToken(sqlParser.ROLE, 0)

        def ROLES(self):
            return self.getToken(sqlParser.ROLES, 0)

        def ROLLBACK(self):
            return self.getToken(sqlParser.ROLLBACK, 0)

        def ROLLUP(self):
            return self.getToken(sqlParser.ROLLUP, 0)

        def ROW(self):
            return self.getToken(sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(sqlParser.ROWS, 0)

        def SCHEMA(self):
            return self.getToken(sqlParser.SCHEMA, 0)

        def SECOND(self):
            return self.getToken(sqlParser.SECOND, 0)

        def SELECT(self):
            return self.getToken(sqlParser.SELECT, 0)

        def SEPARATED(self):
            return self.getToken(sqlParser.SEPARATED, 0)

        def SERDE(self):
            return self.getToken(sqlParser.SERDE, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(sqlParser.SERDEPROPERTIES, 0)

        def SESSION_USER(self):
            return self.getToken(sqlParser.SESSION_USER, 0)

        def SET(self):
            return self.getToken(sqlParser.SET, 0)

        def SETS(self):
            return self.getToken(sqlParser.SETS, 0)

        def SHOW(self):
            return self.getToken(sqlParser.SHOW, 0)

        def SKEWED(self):
            return self.getToken(sqlParser.SKEWED, 0)

        def SOME(self):
            return self.getToken(sqlParser.SOME, 0)

        def SORT(self):
            return self.getToken(sqlParser.SORT, 0)

        def SORTED(self):
            return self.getToken(sqlParser.SORTED, 0)

        def START(self):
            return self.getToken(sqlParser.START, 0)

        def STATISTICS(self):
            return self.getToken(sqlParser.STATISTICS, 0)

        def STORED(self):
            return self.getToken(sqlParser.STORED, 0)

        def STRATIFY(self):
            return self.getToken(sqlParser.STRATIFY, 0)

        def STRUCT(self):
            return self.getToken(sqlParser.STRUCT, 0)

        def SUBSTR(self):
            return self.getToken(sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(sqlParser.SUBSTRING, 0)

        def TABLE(self):
            return self.getToken(sqlParser.TABLE, 0)

        def TABLES(self):
            return self.getToken(sqlParser.TABLES, 0)

        def TABLESAMPLE(self):
            return self.getToken(sqlParser.TABLESAMPLE, 0)

        def TBLPROPERTIES(self):
            return self.getToken(sqlParser.TBLPROPERTIES, 0)

        def TEMPORARY(self):
            return self.getToken(sqlParser.TEMPORARY, 0)

        def TERMINATED(self):
            return self.getToken(sqlParser.TERMINATED, 0)

        def THEN(self):
            return self.getToken(sqlParser.THEN, 0)

        def TO(self):
            return self.getToken(sqlParser.TO, 0)

        def TOUCH(self):
            return self.getToken(sqlParser.TOUCH, 0)

        def TRAILING(self):
            return self.getToken(sqlParser.TRAILING, 0)

        def TRANSACTION(self):
            return self.getToken(sqlParser.TRANSACTION, 0)

        def TRANSACTIONS(self):
            return self.getToken(sqlParser.TRANSACTIONS, 0)

        def TRANSFORM(self):
            return self.getToken(sqlParser.TRANSFORM, 0)

        def TRIM(self):
            return self.getToken(sqlParser.TRIM, 0)

        def TRUE(self):
            return self.getToken(sqlParser.TRUE, 0)

        def TRUNCATE(self):
            return self.getToken(sqlParser.TRUNCATE, 0)

        def TYPE(self):
            return self.getToken(sqlParser.TYPE, 0)

        def UNARCHIVE(self):
            return self.getToken(sqlParser.UNARCHIVE, 0)

        def UNBOUNDED(self):
            return self.getToken(sqlParser.UNBOUNDED, 0)

        def UNCACHE(self):
            return self.getToken(sqlParser.UNCACHE, 0)

        def UNIQUE(self):
            return self.getToken(sqlParser.UNIQUE, 0)

        def UNKNOWN(self):
            return self.getToken(sqlParser.UNKNOWN, 0)

        def UNLOCK(self):
            return self.getToken(sqlParser.UNLOCK, 0)

        def UNSET(self):
            return self.getToken(sqlParser.UNSET, 0)

        def UPDATE(self):
            return self.getToken(sqlParser.UPDATE, 0)

        def USE(self):
            return self.getToken(sqlParser.USE, 0)

        def USER(self):
            return self.getToken(sqlParser.USER, 0)

        def VALUES(self):
            return self.getToken(sqlParser.VALUES, 0)

        def VIEW(self):
            return self.getToken(sqlParser.VIEW, 0)

        def VIEWS(self):
            return self.getToken(sqlParser.VIEWS, 0)

        def WHEN(self):
            return self.getToken(sqlParser.WHEN, 0)

        def WHERE(self):
            return self.getToken(sqlParser.WHERE, 0)

        def WINDOW(self):
            return self.getToken(sqlParser.WINDOW, 0)

        def WITH(self):
            return self.getToken(sqlParser.WITH, 0)

        def YEAR(self):
            return self.getToken(sqlParser.YEAR, 0)

        def getRuleIndex(self):
            return sqlParser.RULE_nonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNonReserved'):
                return visitor.visitNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def nonReserved(self):
        localctx = sqlParser.NonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 272, self.RULE_nonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2994
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & -18014398509748224 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -2382404340318076929 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -545259529 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & 9187343239833681903 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & 2097407 != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    def sempred(self, localctx: RuleContext, ruleIndex: int, predIndex: int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[40] = self.queryTerm_sempred
        self._predicates[94] = self.booleanExpression_sempred
        self._predicates[96] = self.valueExpression_sempred
        self._predicates[97] = self.primaryExpression_sempred
        self._predicates[129] = self.identifier_sempred
        self._predicates[130] = self.strictIdentifier_sempred
        self._predicates[132] = self.number_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception('No predicate with index:' + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def queryTerm_sempred(self, localctx: QueryTermContext, predIndex: int):
        if predIndex == 0:
            return self.precpred(self._ctx, 3)
        if predIndex == 1:
            return self.legacy_setops_precedence_enbled
        if predIndex == 2:
            return self.precpred(self._ctx, 2)
        if predIndex == 3:
            return not self.legacy_setops_precedence_enbled
        if predIndex == 4:
            return self.precpred(self._ctx, 1)
        if predIndex == 5:
            return not self.legacy_setops_precedence_enbled

    def booleanExpression_sempred(self, localctx: BooleanExpressionContext, predIndex: int):
        if predIndex == 6:
            return self.precpred(self._ctx, 2)
        if predIndex == 7:
            return self.precpred(self._ctx, 1)

    def valueExpression_sempred(self, localctx: ValueExpressionContext, predIndex: int):
        if predIndex == 8:
            return self.precpred(self._ctx, 6)
        if predIndex == 9:
            return self.precpred(self._ctx, 5)
        if predIndex == 10:
            return self.precpred(self._ctx, 4)
        if predIndex == 11:
            return self.precpred(self._ctx, 3)
        if predIndex == 12:
            return self.precpred(self._ctx, 2)
        if predIndex == 13:
            return self.precpred(self._ctx, 1)

    def primaryExpression_sempred(self, localctx: PrimaryExpressionContext, predIndex: int):
        if predIndex == 14:
            return self.precpred(self._ctx, 8)
        if predIndex == 15:
            return self.precpred(self._ctx, 6)

    def identifier_sempred(self, localctx: IdentifierContext, predIndex: int):
        if predIndex == 16:
            return not self.SQL_standard_keyword_behavior

    def strictIdentifier_sempred(self, localctx: StrictIdentifierContext, predIndex: int):
        if predIndex == 17:
            return self.SQL_standard_keyword_behavior
        if predIndex == 18:
            return not self.SQL_standard_keyword_behavior

    def number_sempred(self, localctx: NumberContext, predIndex: int):
        if predIndex == 19:
            return not self.legacy_exponent_literal_as_decimal_enabled
        if predIndex == 20:
            return not self.legacy_exponent_literal_as_decimal_enabled
        if predIndex == 21:
            return self.legacy_exponent_literal_as_decimal_enabled