from antlr4 import *
from io import StringIO
import sys
class fugue_sqlParser(Parser):
    grammarFileName = 'fugue_sql.g4'
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
    sharedContextCache = PredictionContextCache()
    literalNames = ['<INVALID>', "'['", "','", "']'", "':'", "'('", "')'", "'.'", "'{'", "'}'", "'true'", "'false'", "'null'", "';'", "'/*+'", "'*/'", "'->'", "'FILL'", "'TAKE'", "'HASH'", "'RAND'", "'EVEN'", "'COARSE'", "'PRESORT'", "'PERSIST'", "'BROADCAST'", "'PARAMS'", "'PROCESS'", "'OUTPUT'", "'OUTTRANSFORM'", "'ROWCOUNT'", "'CONCURRENCY'", "'PREPARTITION'", "'ZIP'", "'PRINT'", "'TITLE'", "'SAVE'", "'APPEND'", "'PARQUET'", "'CSV'", "'JSON'", "'SINGLE'", "'CHECKPOINT'", "'WEAK'", "'STRONG'", "'DETERMINISTIC'", "'YIELD'", "'CONNECT'", "'SAMPLE'", "'SEED'", "'APPROX'", "'SYSTEM'", "'BERNOULLI'", "'RESERVOIR'", "'SUB'", "'CALLBACK'", "'DATAFRAME'", "'FILE'", "'ADD'", "'AFTER'", "'ALL'", "'ALTER'", "'ANALYZE'", "'AND'", "'ANTI'", "'ANY'", "'ARCHIVE'", "'ARRAY'", "'AS'", "'ASC'", "'AT'", "'AUTHORIZATION'", "'BETWEEN'", "'BOTH'", "'BUCKET'", "'BUCKETS'", "'BY'", "'CACHE'", "'CASCADE'", "'CASE'", '<INVALID>', "'CHANGE'", "'CHECK'", "'CLEAR'", "'CLUSTER'", "'CLUSTERED'", "'CODEGEN'", "'COLLATE'", "'COLLECTION'", "'COLUMN'", "'COLUMNS'", "'COMMENT'", "'COMMIT'", "'COMPACT'", "'COMPACTIONS'", "'COMPUTE'", "'CONCATENATE'", "'CONSTRAINT'", "'COST'", "'CREATE'", "'CROSS'", "'CUBE'", "'CURRENT'", "'CURRENT_DATE'", "'CURRENT_TIME'", "'CURRENT_TIMESTAMP'", "'CURRENT_USER'", "'DATA'", "'DATABASE'", '<INVALID>', "'DAY'", "'DBPROPERTIES'", "'DEFINED'", "'DELETE'", "'DELIMITED'", "'DESC'", "'DESCRIBE'", "'DFS'", "'DIRECTORIES'", "'DIRECTORY'", "'DISTINCT'", "'DISTRIBUTE'", "'DROP'", "'ELSE'", "'END'", "'ESCAPE'", "'ESCAPED'", "'EXCEPT'", "'EXCHANGE'", "'EXISTS'", "'EXPLAIN'", "'EXPORT'", "'EXTENDED'", "'EXTERNAL'", "'EXTRACT'", "'FALSE'", "'FETCH'", "'FIELDS'", "'FILTER'", "'FILEFORMAT'", "'FIRST'", "'FOLLOWING'", "'FOR'", "'FOREIGN'", "'FORMAT'", "'FORMATTED'", "'FROM'", "'FULL'", "'FUNCTION'", "'FUNCTIONS'", "'GLOBAL'", "'GRANT'", "'GROUP'", "'GROUPING'", "'HAVING'", "'HOUR'", "'IF'", "'IGNORE'", "'IMPORT'", "'IN'", "'INDEX'", "'INDEXES'", "'INNER'", "'INPATH'", "'INPUTFORMAT'", "'INSERT'", "'INTERSECT'", "'INTERVAL'", "'INTO'", "'IS'", "'ITEMS'", "'JOIN'", "'KEYS'", "'LAST'", "'LATERAL'", "'LAZY'", "'LEADING'", "'LEFT'", "'LIKE'", "'LIMIT'", "'LINES'", "'LIST'", "'LOAD'", "'LOCAL'", "'LOCATION'", "'LOCK'", "'LOCKS'", "'LOGICAL'", "'MACRO'", "'MAP'", "'MATCHED'", "'MERGE'", "'MINUTE'", "'MONTH'", "'MSCK'", "'NAMESPACE'", "'NAMESPACES'", "'NATURAL'", "'NO'", '<INVALID>', "'NULL'", "'NULLS'", "'OF'", "'ON'", "'ONLY'", "'OPTION'", "'OPTIONS'", "'OR'", "'ORDER'", "'OUT'", "'OUTER'", "'OUTPUTFORMAT'", "'OVER'", "'OVERLAPS'", "'OVERLAY'", "'OVERWRITE'", "'PARTITION'", "'PARTITIONED'", "'PARTITIONS'", "'PERCENT'", "'PIVOT'", "'PLACING'", "'POSITION'", "'PRECEDING'", "'PRIMARY'", "'PRINCIPALS'", "'PROPERTIES'", "'PURGE'", "'QUERY'", "'RANGE'", "'RECORDREADER'", "'RECORDWRITER'", "'RECOVER'", "'REDUCE'", "'REFERENCES'", "'REFRESH'", "'RENAME'", "'REPAIR'", "'REPLACE'", "'RESET'", "'RESTRICT'", "'REVOKE'", "'RIGHT'", '<INVALID>', "'ROLE'", "'ROLES'", "'ROLLBACK'", "'ROLLUP'", "'ROW'", "'ROWS'", "'SCHEMA'", "'SECOND'", "'SELECT'", "'SEMI'", "'SEPARATED'", "'SERDE'", "'SERDEPROPERTIES'", "'SESSION_USER'", "'SET'", "'MINUS'", "'SETS'", "'SHOW'", "'SKEWED'", "'SOME'", "'SORT'", "'SORTED'", "'START'", "'STATISTICS'", "'STORED'", "'STRATIFY'", "'STRUCT'", "'SUBSTR'", "'SUBSTRING'", "'TABLE'", "'TABLES'", "'TABLESAMPLE'", "'TBLPROPERTIES'", '<INVALID>', "'TERMINATED'", "'THEN'", "'TO'", "'TOUCH'", "'TRAILING'", "'TRANSACTION'", "'TRANSACTIONS'", "'TRANSFORM'", "'TRIM'", "'TRUE'", "'TRUNCATE'", "'TYPE'", "'UNARCHIVE'", "'UNBOUNDED'", "'UNCACHE'", "'UNION'", "'UNIQUE'", "'UNKNOWN'", "'UNLOCK'", "'UNSET'", "'UPDATE'", "'USE'", "'USER'", "'USING'", "'VALUES'", "'VIEW'", "'VIEWS'", "'WHEN'", "'WHERE'", "'WINDOW'", "'WITH'", "'YEAR'", "'='", "'=='", "'<=>'", "'<>'", "'!='", "'<'", '<INVALID>', "'>'", '<INVALID>', "'+'", "'-'", "'*'", "'/'", "'%'", "'DIV'", "'~'", "'&'", "'|'", "'||'", "'^'"]
    symbolicNames = ['<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', '<INVALID>', 'FILL', 'TAKE', 'HASH', 'RAND', 'EVEN', 'COARSE', 'PRESORT', 'PERSIST', 'BROADCAST', 'PARAMS', 'PROCESS', 'OUTPUT', 'OUTTRANSFORM', 'ROWCOUNT', 'CONCURRENCY', 'PREPARTITION', 'ZIP', 'PRINT', 'TITLE', 'SAVE', 'APPEND', 'PARQUET', 'CSV', 'JSON', 'SINGLE', 'CHECKPOINT', 'WEAK', 'STRONG', 'DETERMINISTIC', 'YIELD', 'CONNECT', 'SAMPLE', 'SEED', 'APPROX', 'SYSTEM', 'BERNOULLI', 'RESERVOIR', 'SUB', 'CALLBACK', 'DATAFRAME', 'FILE', 'ADD', 'AFTER', 'ALL', 'ALTER', 'ANALYZE', 'AND', 'ANTI', 'ANY', 'ARCHIVE', 'ARRAY', 'AS', 'ASC', 'AT', 'AUTHORIZATION', 'BETWEEN', 'BOTH', 'BUCKET', 'BUCKETS', 'BY', 'CACHE', 'CASCADE', 'CASE', 'CAST', 'CHANGE', 'CHECK', 'CLEAR', 'CLUSTER', 'CLUSTERED', 'CODEGEN', 'COLLATE', 'COLLECTION', 'COLUMN', 'COLUMNS', 'COMMENT', 'COMMIT', 'COMPACT', 'COMPACTIONS', 'COMPUTE', 'CONCATENATE', 'CONSTRAINT', 'COST', 'CREATE', 'CROSS', 'CUBE', 'CURRENT', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER', 'DATA', 'DATABASE', 'DATABASES', 'DAY', 'DBPROPERTIES', 'DEFINED', 'DELETE', 'DELIMITED', 'DESC', 'DESCRIBE', 'DFS', 'DIRECTORIES', 'DIRECTORY', 'DISTINCT', 'DISTRIBUTE', 'DROP', 'ELSE', 'END', 'ESCAPE', 'ESCAPED', 'EXCEPT', 'EXCHANGE', 'EXISTS', 'EXPLAIN', 'EXPORT', 'EXTENDED', 'EXTERNAL', 'EXTRACT', 'FALSE', 'FETCH', 'FIELDS', 'FILTER', 'FILEFORMAT', 'FIRST', 'FOLLOWING', 'FOR', 'FOREIGN', 'FORMAT', 'FORMATTED', 'FROM', 'FULL', 'FUNCTION', 'FUNCTIONS', 'GLOBAL', 'GRANT', 'GROUP', 'GROUPING', 'HAVING', 'HOUR', 'IF', 'IGNORE', 'IMPORT', 'IN', 'INDEX', 'INDEXES', 'INNER', 'INPATH', 'INPUTFORMAT', 'INSERT', 'INTERSECT', 'INTERVAL', 'INTO', 'IS', 'ITEMS', 'JOIN', 'KEYS', 'LAST', 'LATERAL', 'LAZY', 'LEADING', 'LEFT', 'LIKE', 'LIMIT', 'LINES', 'LIST', 'LOAD', 'LOCAL', 'LOCATION', 'LOCK', 'LOCKS', 'LOGICAL', 'MACRO', 'MAP', 'MATCHED', 'MERGE', 'MINUTE', 'MONTH', 'MSCK', 'NAMESPACE', 'NAMESPACES', 'NATURAL', 'NO', 'NOT', 'THENULL', 'THENULLS', 'OF', 'ON', 'ONLY', 'OPTION', 'OPTIONS', 'OR', 'ORDER', 'OUT', 'OUTER', 'OUTPUTFORMAT', 'OVER', 'OVERLAPS', 'OVERLAY', 'OVERWRITE', 'PARTITION', 'PARTITIONED', 'PARTITIONS', 'PERCENTLIT', 'PIVOT', 'PLACING', 'POSITION', 'PRECEDING', 'PRIMARY', 'PRINCIPALS', 'PROPERTIES', 'PURGE', 'QUERY', 'RANGE', 'RECORDREADER', 'RECORDWRITER', 'RECOVER', 'REDUCE', 'REFERENCES', 'REFRESH', 'RENAME', 'REPAIR', 'REPLACE', 'RESET', 'RESTRICT', 'REVOKE', 'RIGHT', 'RLIKE', 'ROLE', 'ROLES', 'ROLLBACK', 'ROLLUP', 'ROW', 'ROWS', 'SCHEMA', 'SECOND', 'SELECT', 'SEMI', 'SEPARATED', 'SERDE', 'SERDEPROPERTIES', 'SESSION_USER', 'SET', 'SETMINUS', 'SETS', 'SHOW', 'SKEWED', 'SOME', 'SORT', 'SORTED', 'START', 'STATISTICS', 'STORED', 'STRATIFY', 'STRUCT', 'SUBSTR', 'SUBSTRING', 'TABLE', 'TABLES', 'TABLESAMPLE', 'TBLPROPERTIES', 'TEMPORARY', 'TERMINATED', 'THEN', 'TO', 'TOUCH', 'TRAILING', 'TRANSACTION', 'TRANSACTIONS', 'TRANSFORM', 'TRIM', 'TRUE', 'TRUNCATE', 'TYPE', 'UNARCHIVE', 'UNBOUNDED', 'UNCACHE', 'UNION', 'UNIQUE', 'UNKNOWN', 'UNLOCK', 'UNSET', 'UPDATE', 'USE', 'USER', 'USING', 'VALUES', 'VIEW', 'VIEWS', 'WHEN', 'WHERE', 'WINDOW', 'WITH', 'YEAR', 'EQUAL', 'DOUBLEEQUAL', 'NSEQ', 'NEQ', 'NEQJ', 'LT', 'LTE', 'GT', 'GTE', 'PLUS', 'MINUS', 'ASTERISK', 'SLASH', 'PERCENT', 'DIV', 'TILDE', 'AMPERSAND', 'PIPE', 'CONCAT_PIPE', 'HAT', 'STRING', 'BIGINT_LITERAL', 'SMALLINT_LITERAL', 'TINYINT_LITERAL', 'INTEGER_VALUE', 'EXPONENT_VALUE', 'DECIMAL_VALUE', 'DOUBLE_LITERAL', 'BIGDECIMAL_LITERAL', 'IDENTIFIER', 'BACKQUOTED_IDENTIFIER', 'SIMPLE_COMMENT', 'BRACKETED_COMMENT', 'WS', 'UNRECOGNIZED']
    RULE_fugueLanguage = 0
    RULE_fugueSingleStatement = 1
    RULE_fugueSingleTask = 2
    RULE_fugueNestableTask = 3
    RULE_fugueNestableTaskCollectionNoSelect = 4
    RULE_fugueTransformTask = 5
    RULE_fugueProcessTask = 6
    RULE_fugueSaveAndUseTask = 7
    RULE_fugueRenameColumnsTask = 8
    RULE_fugueAlterColumnsTask = 9
    RULE_fugueDropColumnsTask = 10
    RULE_fugueDropnaTask = 11
    RULE_fugueFillnaTask = 12
    RULE_fugueSampleTask = 13
    RULE_fugueTakeTask = 14
    RULE_fugueZipTask = 15
    RULE_fugueCreateTask = 16
    RULE_fugueCreateDataTask = 17
    RULE_fugueLoadTask = 18
    RULE_fugueOutputTask = 19
    RULE_fuguePrintTask = 20
    RULE_fugueSaveTask = 21
    RULE_fugueOutputTransformTask = 22
    RULE_fugueModuleTask = 23
    RULE_fugueSqlEngine = 24
    RULE_fugueSingleFile = 25
    RULE_fugueLoadColumns = 26
    RULE_fugueSaveMode = 27
    RULE_fugueFileFormat = 28
    RULE_fuguePath = 29
    RULE_fuguePaths = 30
    RULE_fugueCheckpoint = 31
    RULE_fugueCheckpointNamespace = 32
    RULE_fugueYield = 33
    RULE_fugueBroadcast = 34
    RULE_fugueDataFrames = 35
    RULE_fugueDataFramePair = 36
    RULE_fugueDataFrame = 37
    RULE_fugueDataFrameMember = 38
    RULE_fugueAssignment = 39
    RULE_fugueAssignmentSign = 40
    RULE_fugueSingleOutputExtensionCommonWild = 41
    RULE_fugueSingleOutputExtensionCommon = 42
    RULE_fugueExtension = 43
    RULE_fugueSampleMethod = 44
    RULE_fugueZipType = 45
    RULE_fuguePrepartition = 46
    RULE_fuguePartitionAlgo = 47
    RULE_fuguePartitionNum = 48
    RULE_fuguePartitionNumber = 49
    RULE_fugueParams = 50
    RULE_fugueCols = 51
    RULE_fugueColsSort = 52
    RULE_fugueColSort = 53
    RULE_fugueColumnIdentifier = 54
    RULE_fugueRenameExpression = 55
    RULE_fugueWildSchema = 56
    RULE_fugueWildSchemaPair = 57
    RULE_fugueSchemaOp = 58
    RULE_fugueSchema = 59
    RULE_fugueSchemaPair = 60
    RULE_fugueSchemaKey = 61
    RULE_fugueSchemaType = 62
    RULE_fugueRenamePair = 63
    RULE_fugueJson = 64
    RULE_fugueJsonObj = 65
    RULE_fugueJsonPairs = 66
    RULE_fugueJsonPair = 67
    RULE_fugueJsonKey = 68
    RULE_fugueJsonArray = 69
    RULE_fugueJsonValue = 70
    RULE_fugueJsonNumber = 71
    RULE_fugueJsonString = 72
    RULE_fugueJsonBool = 73
    RULE_fugueJsonNull = 74
    RULE_fugueIdentifier = 75
    RULE_singleStatement = 76
    RULE_singleExpression = 77
    RULE_singleTableIdentifier = 78
    RULE_singleMultipartIdentifier = 79
    RULE_singleFunctionIdentifier = 80
    RULE_singleDataType = 81
    RULE_singleTableSchema = 82
    RULE_statement = 83
    RULE_unsupportedHiveNativeCommands = 84
    RULE_createTableHeader = 85
    RULE_replaceTableHeader = 86
    RULE_bucketSpec = 87
    RULE_skewSpec = 88
    RULE_locationSpec = 89
    RULE_commentSpec = 90
    RULE_query = 91
    RULE_insertInto = 92
    RULE_partitionSpecLocation = 93
    RULE_partitionSpec = 94
    RULE_partitionVal = 95
    RULE_theNamespace = 96
    RULE_describeFuncName = 97
    RULE_describeColName = 98
    RULE_ctes = 99
    RULE_namedQuery = 100
    RULE_tableProvider = 101
    RULE_createTableClauses = 102
    RULE_tablePropertyList = 103
    RULE_tableProperty = 104
    RULE_tablePropertyKey = 105
    RULE_tablePropertyValue = 106
    RULE_constantList = 107
    RULE_nestedConstantList = 108
    RULE_createFileFormat = 109
    RULE_fileFormat = 110
    RULE_storageHandler = 111
    RULE_resource = 112
    RULE_dmlStatementNoWith = 113
    RULE_queryOrganization = 114
    RULE_multiInsertQueryBody = 115
    RULE_queryTerm = 116
    RULE_queryPrimary = 117
    RULE_sortItem = 118
    RULE_fromStatement = 119
    RULE_fromStatementBody = 120
    RULE_querySpecification = 121
    RULE_optionalFromClause = 122
    RULE_transformClause = 123
    RULE_selectClause = 124
    RULE_setClause = 125
    RULE_matchedClause = 126
    RULE_notMatchedClause = 127
    RULE_matchedAction = 128
    RULE_notMatchedAction = 129
    RULE_assignmentList = 130
    RULE_assignment = 131
    RULE_whereClause = 132
    RULE_havingClause = 133
    RULE_hint = 134
    RULE_hintStatement = 135
    RULE_fromClause = 136
    RULE_aggregationClause = 137
    RULE_groupingSet = 138
    RULE_pivotClause = 139
    RULE_pivotColumn = 140
    RULE_pivotValue = 141
    RULE_lateralView = 142
    RULE_setQuantifier = 143
    RULE_relation = 144
    RULE_joinRelation = 145
    RULE_joinType = 146
    RULE_joinCriteria = 147
    RULE_sample = 148
    RULE_sampleMethod = 149
    RULE_identifierList = 150
    RULE_identifierSeq = 151
    RULE_orderedIdentifierList = 152
    RULE_orderedIdentifier = 153
    RULE_identifierCommentList = 154
    RULE_identifierComment = 155
    RULE_relationPrimary = 156
    RULE_inlineTable = 157
    RULE_functionTable = 158
    RULE_tableAlias = 159
    RULE_rowFormat = 160
    RULE_multipartIdentifierList = 161
    RULE_multipartIdentifier = 162
    RULE_tableIdentifier = 163
    RULE_functionIdentifier = 164
    RULE_namedExpression = 165
    RULE_namedExpressionSeq = 166
    RULE_transformList = 167
    RULE_transform = 168
    RULE_transformArgument = 169
    RULE_expression = 170
    RULE_booleanExpression = 171
    RULE_predicate = 172
    RULE_valueExpression = 173
    RULE_primaryExpression = 174
    RULE_constant = 175
    RULE_comparisonOperator = 176
    RULE_comparisonEqualOperator = 177
    RULE_arithmeticOperator = 178
    RULE_predicateOperator = 179
    RULE_booleanValue = 180
    RULE_interval = 181
    RULE_errorCapturingMultiUnitsInterval = 182
    RULE_multiUnitsInterval = 183
    RULE_errorCapturingUnitToUnitInterval = 184
    RULE_unitToUnitInterval = 185
    RULE_intervalValue = 186
    RULE_intervalUnit = 187
    RULE_colPosition = 188
    RULE_dataType = 189
    RULE_qualifiedColTypeWithPositionList = 190
    RULE_qualifiedColTypeWithPosition = 191
    RULE_colTypeList = 192
    RULE_colType = 193
    RULE_complexColTypeList = 194
    RULE_complexColType = 195
    RULE_whenClause = 196
    RULE_windowClause = 197
    RULE_namedWindow = 198
    RULE_windowSpec = 199
    RULE_windowFrame = 200
    RULE_frameBound = 201
    RULE_qualifiedNameList = 202
    RULE_functionName = 203
    RULE_qualifiedName = 204
    RULE_errorCapturingIdentifier = 205
    RULE_errorCapturingIdentifierExtra = 206
    RULE_identifier = 207
    RULE_strictIdentifier = 208
    RULE_quotedIdentifier = 209
    RULE_number = 210
    RULE_alterColumnAction = 211
    RULE_ansiNonReserved = 212
    RULE_strictNonReserved = 213
    RULE_nonReserved = 214
    ruleNames = ['fugueLanguage', 'fugueSingleStatement', 'fugueSingleTask', 'fugueNestableTask', 'fugueNestableTaskCollectionNoSelect', 'fugueTransformTask', 'fugueProcessTask', 'fugueSaveAndUseTask', 'fugueRenameColumnsTask', 'fugueAlterColumnsTask', 'fugueDropColumnsTask', 'fugueDropnaTask', 'fugueFillnaTask', 'fugueSampleTask', 'fugueTakeTask', 'fugueZipTask', 'fugueCreateTask', 'fugueCreateDataTask', 'fugueLoadTask', 'fugueOutputTask', 'fuguePrintTask', 'fugueSaveTask', 'fugueOutputTransformTask', 'fugueModuleTask', 'fugueSqlEngine', 'fugueSingleFile', 'fugueLoadColumns', 'fugueSaveMode', 'fugueFileFormat', 'fuguePath', 'fuguePaths', 'fugueCheckpoint', 'fugueCheckpointNamespace', 'fugueYield', 'fugueBroadcast', 'fugueDataFrames', 'fugueDataFramePair', 'fugueDataFrame', 'fugueDataFrameMember', 'fugueAssignment', 'fugueAssignmentSign', 'fugueSingleOutputExtensionCommonWild', 'fugueSingleOutputExtensionCommon', 'fugueExtension', 'fugueSampleMethod', 'fugueZipType', 'fuguePrepartition', 'fuguePartitionAlgo', 'fuguePartitionNum', 'fuguePartitionNumber', 'fugueParams', 'fugueCols', 'fugueColsSort', 'fugueColSort', 'fugueColumnIdentifier', 'fugueRenameExpression', 'fugueWildSchema', 'fugueWildSchemaPair', 'fugueSchemaOp', 'fugueSchema', 'fugueSchemaPair', 'fugueSchemaKey', 'fugueSchemaType', 'fugueRenamePair', 'fugueJson', 'fugueJsonObj', 'fugueJsonPairs', 'fugueJsonPair', 'fugueJsonKey', 'fugueJsonArray', 'fugueJsonValue', 'fugueJsonNumber', 'fugueJsonString', 'fugueJsonBool', 'fugueJsonNull', 'fugueIdentifier', 'singleStatement', 'singleExpression', 'singleTableIdentifier', 'singleMultipartIdentifier', 'singleFunctionIdentifier', 'singleDataType', 'singleTableSchema', 'statement', 'unsupportedHiveNativeCommands', 'createTableHeader', 'replaceTableHeader', 'bucketSpec', 'skewSpec', 'locationSpec', 'commentSpec', 'query', 'insertInto', 'partitionSpecLocation', 'partitionSpec', 'partitionVal', 'theNamespace', 'describeFuncName', 'describeColName', 'ctes', 'namedQuery', 'tableProvider', 'createTableClauses', 'tablePropertyList', 'tableProperty', 'tablePropertyKey', 'tablePropertyValue', 'constantList', 'nestedConstantList', 'createFileFormat', 'fileFormat', 'storageHandler', 'resource', 'dmlStatementNoWith', 'queryOrganization', 'multiInsertQueryBody', 'queryTerm', 'queryPrimary', 'sortItem', 'fromStatement', 'fromStatementBody', 'querySpecification', 'optionalFromClause', 'transformClause', 'selectClause', 'setClause', 'matchedClause', 'notMatchedClause', 'matchedAction', 'notMatchedAction', 'assignmentList', 'assignment', 'whereClause', 'havingClause', 'hint', 'hintStatement', 'fromClause', 'aggregationClause', 'groupingSet', 'pivotClause', 'pivotColumn', 'pivotValue', 'lateralView', 'setQuantifier', 'relation', 'joinRelation', 'joinType', 'joinCriteria', 'sample', 'sampleMethod', 'identifierList', 'identifierSeq', 'orderedIdentifierList', 'orderedIdentifier', 'identifierCommentList', 'identifierComment', 'relationPrimary', 'inlineTable', 'functionTable', 'tableAlias', 'rowFormat', 'multipartIdentifierList', 'multipartIdentifier', 'tableIdentifier', 'functionIdentifier', 'namedExpression', 'namedExpressionSeq', 'transformList', 'transform', 'transformArgument', 'expression', 'booleanExpression', 'predicate', 'valueExpression', 'primaryExpression', 'constant', 'comparisonOperator', 'comparisonEqualOperator', 'arithmeticOperator', 'predicateOperator', 'booleanValue', 'interval', 'errorCapturingMultiUnitsInterval', 'multiUnitsInterval', 'errorCapturingUnitToUnitInterval', 'unitToUnitInterval', 'intervalValue', 'intervalUnit', 'colPosition', 'dataType', 'qualifiedColTypeWithPositionList', 'qualifiedColTypeWithPosition', 'colTypeList', 'colType', 'complexColTypeList', 'complexColType', 'whenClause', 'windowClause', 'namedWindow', 'windowSpec', 'windowFrame', 'frameBound', 'qualifiedNameList', 'functionName', 'qualifiedName', 'errorCapturingIdentifier', 'errorCapturingIdentifierExtra', 'identifier', 'strictIdentifier', 'quotedIdentifier', 'number', 'alterColumnAction', 'ansiNonReserved', 'strictNonReserved', 'nonReserved']
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
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    FILL = 17
    TAKE = 18
    HASH = 19
    RAND = 20
    EVEN = 21
    COARSE = 22
    PRESORT = 23
    PERSIST = 24
    BROADCAST = 25
    PARAMS = 26
    PROCESS = 27
    OUTPUT = 28
    OUTTRANSFORM = 29
    ROWCOUNT = 30
    CONCURRENCY = 31
    PREPARTITION = 32
    ZIP = 33
    PRINT = 34
    TITLE = 35
    SAVE = 36
    APPEND = 37
    PARQUET = 38
    CSV = 39
    JSON = 40
    SINGLE = 41
    CHECKPOINT = 42
    WEAK = 43
    STRONG = 44
    DETERMINISTIC = 45
    YIELD = 46
    CONNECT = 47
    SAMPLE = 48
    SEED = 49
    APPROX = 50
    SYSTEM = 51
    BERNOULLI = 52
    RESERVOIR = 53
    SUB = 54
    CALLBACK = 55
    DATAFRAME = 56
    FILE = 57
    ADD = 58
    AFTER = 59
    ALL = 60
    ALTER = 61
    ANALYZE = 62
    AND = 63
    ANTI = 64
    ANY = 65
    ARCHIVE = 66
    ARRAY = 67
    AS = 68
    ASC = 69
    AT = 70
    AUTHORIZATION = 71
    BETWEEN = 72
    BOTH = 73
    BUCKET = 74
    BUCKETS = 75
    BY = 76
    CACHE = 77
    CASCADE = 78
    CASE = 79
    CAST = 80
    CHANGE = 81
    CHECK = 82
    CLEAR = 83
    CLUSTER = 84
    CLUSTERED = 85
    CODEGEN = 86
    COLLATE = 87
    COLLECTION = 88
    COLUMN = 89
    COLUMNS = 90
    COMMENT = 91
    COMMIT = 92
    COMPACT = 93
    COMPACTIONS = 94
    COMPUTE = 95
    CONCATENATE = 96
    CONSTRAINT = 97
    COST = 98
    CREATE = 99
    CROSS = 100
    CUBE = 101
    CURRENT = 102
    CURRENT_DATE = 103
    CURRENT_TIME = 104
    CURRENT_TIMESTAMP = 105
    CURRENT_USER = 106
    DATA = 107
    DATABASE = 108
    DATABASES = 109
    DAY = 110
    DBPROPERTIES = 111
    DEFINED = 112
    DELETE = 113
    DELIMITED = 114
    DESC = 115
    DESCRIBE = 116
    DFS = 117
    DIRECTORIES = 118
    DIRECTORY = 119
    DISTINCT = 120
    DISTRIBUTE = 121
    DROP = 122
    ELSE = 123
    END = 124
    ESCAPE = 125
    ESCAPED = 126
    EXCEPT = 127
    EXCHANGE = 128
    EXISTS = 129
    EXPLAIN = 130
    EXPORT = 131
    EXTENDED = 132
    EXTERNAL = 133
    EXTRACT = 134
    FALSE = 135
    FETCH = 136
    FIELDS = 137
    FILTER = 138
    FILEFORMAT = 139
    FIRST = 140
    FOLLOWING = 141
    FOR = 142
    FOREIGN = 143
    FORMAT = 144
    FORMATTED = 145
    FROM = 146
    FULL = 147
    FUNCTION = 148
    FUNCTIONS = 149
    GLOBAL = 150
    GRANT = 151
    GROUP = 152
    GROUPING = 153
    HAVING = 154
    HOUR = 155
    IF = 156
    IGNORE = 157
    IMPORT = 158
    IN = 159
    INDEX = 160
    INDEXES = 161
    INNER = 162
    INPATH = 163
    INPUTFORMAT = 164
    INSERT = 165
    INTERSECT = 166
    INTERVAL = 167
    INTO = 168
    IS = 169
    ITEMS = 170
    JOIN = 171
    KEYS = 172
    LAST = 173
    LATERAL = 174
    LAZY = 175
    LEADING = 176
    LEFT = 177
    LIKE = 178
    LIMIT = 179
    LINES = 180
    LIST = 181
    LOAD = 182
    LOCAL = 183
    LOCATION = 184
    LOCK = 185
    LOCKS = 186
    LOGICAL = 187
    MACRO = 188
    MAP = 189
    MATCHED = 190
    MERGE = 191
    MINUTE = 192
    MONTH = 193
    MSCK = 194
    NAMESPACE = 195
    NAMESPACES = 196
    NATURAL = 197
    NO = 198
    NOT = 199
    THENULL = 200
    THENULLS = 201
    OF = 202
    ON = 203
    ONLY = 204
    OPTION = 205
    OPTIONS = 206
    OR = 207
    ORDER = 208
    OUT = 209
    OUTER = 210
    OUTPUTFORMAT = 211
    OVER = 212
    OVERLAPS = 213
    OVERLAY = 214
    OVERWRITE = 215
    PARTITION = 216
    PARTITIONED = 217
    PARTITIONS = 218
    PERCENTLIT = 219
    PIVOT = 220
    PLACING = 221
    POSITION = 222
    PRECEDING = 223
    PRIMARY = 224
    PRINCIPALS = 225
    PROPERTIES = 226
    PURGE = 227
    QUERY = 228
    RANGE = 229
    RECORDREADER = 230
    RECORDWRITER = 231
    RECOVER = 232
    REDUCE = 233
    REFERENCES = 234
    REFRESH = 235
    RENAME = 236
    REPAIR = 237
    REPLACE = 238
    RESET = 239
    RESTRICT = 240
    REVOKE = 241
    RIGHT = 242
    RLIKE = 243
    ROLE = 244
    ROLES = 245
    ROLLBACK = 246
    ROLLUP = 247
    ROW = 248
    ROWS = 249
    SCHEMA = 250
    SECOND = 251
    SELECT = 252
    SEMI = 253
    SEPARATED = 254
    SERDE = 255
    SERDEPROPERTIES = 256
    SESSION_USER = 257
    SET = 258
    SETMINUS = 259
    SETS = 260
    SHOW = 261
    SKEWED = 262
    SOME = 263
    SORT = 264
    SORTED = 265
    START = 266
    STATISTICS = 267
    STORED = 268
    STRATIFY = 269
    STRUCT = 270
    SUBSTR = 271
    SUBSTRING = 272
    TABLE = 273
    TABLES = 274
    TABLESAMPLE = 275
    TBLPROPERTIES = 276
    TEMPORARY = 277
    TERMINATED = 278
    THEN = 279
    TO = 280
    TOUCH = 281
    TRAILING = 282
    TRANSACTION = 283
    TRANSACTIONS = 284
    TRANSFORM = 285
    TRIM = 286
    TRUE = 287
    TRUNCATE = 288
    TYPE = 289
    UNARCHIVE = 290
    UNBOUNDED = 291
    UNCACHE = 292
    UNION = 293
    UNIQUE = 294
    UNKNOWN = 295
    UNLOCK = 296
    UNSET = 297
    UPDATE = 298
    USE = 299
    USER = 300
    USING = 301
    VALUES = 302
    VIEW = 303
    VIEWS = 304
    WHEN = 305
    WHERE = 306
    WINDOW = 307
    WITH = 308
    YEAR = 309
    EQUAL = 310
    DOUBLEEQUAL = 311
    NSEQ = 312
    NEQ = 313
    NEQJ = 314
    LT = 315
    LTE = 316
    GT = 317
    GTE = 318
    PLUS = 319
    MINUS = 320
    ASTERISK = 321
    SLASH = 322
    PERCENT = 323
    DIV = 324
    TILDE = 325
    AMPERSAND = 326
    PIPE = 327
    CONCAT_PIPE = 328
    HAT = 329
    STRING = 330
    BIGINT_LITERAL = 331
    SMALLINT_LITERAL = 332
    TINYINT_LITERAL = 333
    INTEGER_VALUE = 334
    EXPONENT_VALUE = 335
    DECIMAL_VALUE = 336
    DOUBLE_LITERAL = 337
    BIGDECIMAL_LITERAL = 338
    IDENTIFIER = 339
    BACKQUOTED_IDENTIFIER = 340
    SIMPLE_COMMENT = 341
    BRACKETED_COMMENT = 342
    WS = 343
    UNRECOGNIZED = 344

    def __init__(self, input: TokenStream, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.11.1')
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None

    class FugueLanguageContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def fugueSingleTask(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSingleTaskContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSingleTaskContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueLanguage

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueLanguage'):
                return visitor.visitFugueLanguage(self)
            else:
                return visitor.visitChildren(self)

    def fugueLanguage(self):
        localctx = fugue_sqlParser.FugueLanguageContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_fugueLanguage)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 431
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 430
                self.fugueSingleTask()
                self.state = 433
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la & ~63 == 0 and 1 << _la & -269793669747965952 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & 18014398509481983 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0)):
                    break
            self.state = 435
            self.match(fugue_sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSingleStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueSingleTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleTaskContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSingleStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSingleStatement'):
                return visitor.visitFugueSingleStatement(self)
            else:
                return visitor.visitChildren(self)

    def fugueSingleStatement(self):
        localctx = fugue_sqlParser.FugueSingleStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_fugueSingleStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 437
            self.fugueSingleTask()
            self.state = 438
            self.match(fugue_sqlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSingleTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueNestableTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskContext, 0)

        def fugueOutputTransformTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueOutputTransformTaskContext, 0)

        def fugueOutputTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueOutputTaskContext, 0)

        def fuguePrintTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrintTaskContext, 0)

        def fugueSaveTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSaveTaskContext, 0)

        def fugueModuleTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueModuleTaskContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSingleTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSingleTask'):
                return visitor.visitFugueSingleTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueSingleTask(self):
        localctx = fugue_sqlParser.FugueSingleTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_fugueSingleTask)
        try:
            self.state = 446
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 1, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 440
                self.fugueNestableTask()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 441
                self.fugueOutputTransformTask()
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 442
                self.fugueOutputTask()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 443
                self.fuguePrintTask()
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 444
                self.fugueSaveTask()
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 445
                self.fugueModuleTask()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueNestableTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.assign = None
            self.q = None
            self.checkpoint = None
            self.broadcast = None
            self.y = None

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def fugueAssignment(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentContext, 0)

        def fugueCheckpoint(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueCheckpointContext, 0)

        def fugueBroadcast(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueBroadcastContext, 0)

        def fugueYield(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueYieldContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueNestableTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueNestableTask'):
                return visitor.visitFugueNestableTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueNestableTask(self):
        localctx = fugue_sqlParser.FugueNestableTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_fugueNestableTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 449
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 2, self._ctx)
            if la_ == 1:
                self.state = 448
                localctx.assign = self.fugueAssignment()
            self.state = 451
            localctx.q = self.query()
            self.state = 453
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 3, self._ctx)
            if la_ == 1:
                self.state = 452
                localctx.checkpoint = self.fugueCheckpoint()
            self.state = 456
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 25:
                self.state = 455
                localctx.broadcast = self.fugueBroadcast()
            self.state = 459
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 46:
                self.state = 458
                localctx.y = self.fugueYield()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueNestableTaskCollectionNoSelectContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueTransformTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueTransformTaskContext, 0)

        def fugueProcessTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueProcessTaskContext, 0)

        def fugueZipTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueZipTaskContext, 0)

        def fugueCreateTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueCreateTaskContext, 0)

        def fugueCreateDataTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueCreateDataTaskContext, 0)

        def fugueLoadTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueLoadTaskContext, 0)

        def fugueSaveAndUseTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSaveAndUseTaskContext, 0)

        def fugueRenameColumnsTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueRenameColumnsTaskContext, 0)

        def fugueAlterColumnsTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueAlterColumnsTaskContext, 0)

        def fugueDropColumnsTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDropColumnsTaskContext, 0)

        def fugueDropnaTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDropnaTaskContext, 0)

        def fugueFillnaTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueFillnaTaskContext, 0)

        def fugueSampleTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSampleTaskContext, 0)

        def fugueTakeTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueTakeTaskContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueNestableTaskCollectionNoSelect

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueNestableTaskCollectionNoSelect'):
                return visitor.visitFugueNestableTaskCollectionNoSelect(self)
            else:
                return visitor.visitChildren(self)

    def fugueNestableTaskCollectionNoSelect(self):
        localctx = fugue_sqlParser.FugueNestableTaskCollectionNoSelectContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_fugueNestableTaskCollectionNoSelect)
        try:
            self.state = 475
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 6, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 461
                self.fugueTransformTask()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 462
                self.fugueProcessTask()
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 463
                self.fugueZipTask()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 464
                self.fugueCreateTask()
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 465
                self.fugueCreateDataTask()
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 466
                self.fugueLoadTask()
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 467
                self.fugueSaveAndUseTask()
                pass
            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 468
                self.fugueRenameColumnsTask()
                pass
            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 469
                self.fugueAlterColumnsTask()
                pass
            elif la_ == 10:
                self.enterOuterAlt(localctx, 10)
                self.state = 470
                self.fugueDropColumnsTask()
                pass
            elif la_ == 11:
                self.enterOuterAlt(localctx, 11)
                self.state = 471
                self.fugueDropnaTask()
                pass
            elif la_ == 12:
                self.enterOuterAlt(localctx, 12)
                self.state = 472
                self.fugueFillnaTask()
                pass
            elif la_ == 13:
                self.enterOuterAlt(localctx, 13)
                self.state = 473
                self.fugueSampleTask()
                pass
            elif la_ == 14:
                self.enterOuterAlt(localctx, 14)
                self.state = 474
                self.fugueTakeTask()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueTransformTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.dfs = None
            self.partition = None
            self.params = None
            self.callback = None

        def TRANSFORM(self):
            return self.getToken(fugue_sqlParser.TRANSFORM, 0)

        def fugueSingleOutputExtensionCommonWild(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonWildContext, 0)

        def CALLBACK(self):
            return self.getToken(fugue_sqlParser.CALLBACK, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueTransformTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueTransformTask'):
                return visitor.visitFugueTransformTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueTransformTask(self):
        localctx = fugue_sqlParser.FugueTransformTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_fugueTransformTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 477
            self.match(fugue_sqlParser.TRANSFORM)
            self.state = 479
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 7, self._ctx)
            if la_ == 1:
                self.state = 478
                localctx.dfs = self.fugueDataFrames()
            self.state = 482
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 481
                localctx.partition = self.fuguePrepartition()
            self.state = 484
            localctx.params = self.fugueSingleOutputExtensionCommonWild()
            self.state = 487
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 9, self._ctx)
            if la_ == 1:
                self.state = 485
                self.match(fugue_sqlParser.CALLBACK)
                self.state = 486
                localctx.callback = self.fugueExtension()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueProcessTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.dfs = None
            self.partition = None
            self.params = None

        def PROCESS(self):
            return self.getToken(fugue_sqlParser.PROCESS, 0)

        def fugueSingleOutputExtensionCommon(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonContext, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueProcessTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueProcessTask'):
                return visitor.visitFugueProcessTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueProcessTask(self):
        localctx = fugue_sqlParser.FugueProcessTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_fugueProcessTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 489
            self.match(fugue_sqlParser.PROCESS)
            self.state = 491
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 10, self._ctx)
            if la_ == 1:
                self.state = 490
                localctx.dfs = self.fugueDataFrames()
            self.state = 494
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 493
                localctx.partition = self.fuguePrepartition()
            self.state = 496
            localctx.params = self.fugueSingleOutputExtensionCommon()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSaveAndUseTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.df = None
            self.partition = None
            self.m = None
            self.single = None
            self.fmt = None
            self.path = None
            self.params = None

        def SAVE(self):
            return self.getToken(fugue_sqlParser.SAVE, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def USE(self):
            return self.getToken(fugue_sqlParser.USE, 0)

        def fugueSaveMode(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSaveModeContext, 0)

        def fuguePath(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueSingleFile(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

        def fugueFileFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSaveAndUseTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSaveAndUseTask'):
                return visitor.visitFugueSaveAndUseTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueSaveAndUseTask(self):
        localctx = fugue_sqlParser.FugueSaveAndUseTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_fugueSaveAndUseTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 498
            self.match(fugue_sqlParser.SAVE)
            self.state = 499
            self.match(fugue_sqlParser.AND)
            self.state = 500
            self.match(fugue_sqlParser.USE)
            self.state = 502
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 12, self._ctx)
            if la_ == 1:
                self.state = 501
                localctx.df = self.fugueDataFrame()
            self.state = 505
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 504
                localctx.partition = self.fuguePrepartition()
            self.state = 507
            localctx.m = self.fugueSaveMode()
            self.state = 509
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 41:
                self.state = 508
                localctx.single = self.fugueSingleFile()
            self.state = 512
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 1924145348608 != 0:
                self.state = 511
                localctx.fmt = self.fugueFileFormat()
            self.state = 514
            localctx.path = self.fuguePath()
            self.state = 516
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 16, self._ctx)
            if la_ == 1:
                self.state = 515
                localctx.params = self.fugueParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueRenameColumnsTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.cols = None
            self.df = None

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def fugueRenameExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueRenameExpressionContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueRenameColumnsTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueRenameColumnsTask'):
                return visitor.visitFugueRenameColumnsTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueRenameColumnsTask(self):
        localctx = fugue_sqlParser.FugueRenameColumnsTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_fugueRenameColumnsTask)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 518
            self.match(fugue_sqlParser.RENAME)
            self.state = 519
            self.match(fugue_sqlParser.COLUMNS)
            self.state = 520
            localctx.cols = self.fugueRenameExpression()
            self.state = 523
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 17, self._ctx)
            if la_ == 1:
                self.state = 521
                self.match(fugue_sqlParser.FROM)
                self.state = 522
                localctx.df = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueAlterColumnsTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.cols = None
            self.df = None

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueAlterColumnsTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueAlterColumnsTask'):
                return visitor.visitFugueAlterColumnsTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueAlterColumnsTask(self):
        localctx = fugue_sqlParser.FugueAlterColumnsTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_fugueAlterColumnsTask)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 525
            self.match(fugue_sqlParser.ALTER)
            self.state = 526
            self.match(fugue_sqlParser.COLUMNS)
            self.state = 527
            localctx.cols = self.fugueSchema()
            self.state = 530
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 18, self._ctx)
            if la_ == 1:
                self.state = 528
                self.match(fugue_sqlParser.FROM)
                self.state = 529
                localctx.df = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueDropColumnsTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.cols = None
            self.df = None

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDropColumnsTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDropColumnsTask'):
                return visitor.visitFugueDropColumnsTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueDropColumnsTask(self):
        localctx = fugue_sqlParser.FugueDropColumnsTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_fugueDropColumnsTask)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 532
            self.match(fugue_sqlParser.DROP)
            self.state = 533
            self.match(fugue_sqlParser.COLUMNS)
            self.state = 534
            localctx.cols = self.fugueCols()
            self.state = 537
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 19, self._ctx)
            if la_ == 1:
                self.state = 535
                self.match(fugue_sqlParser.IF)
                self.state = 536
                self.match(fugue_sqlParser.EXISTS)
            self.state = 541
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 20, self._ctx)
            if la_ == 1:
                self.state = 539
                self.match(fugue_sqlParser.FROM)
                self.state = 540
                localctx.df = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueDropnaTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.how = None
            self.cols = None
            self.df = None

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

        def ANY(self):
            return self.getToken(fugue_sqlParser.ANY, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDropnaTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDropnaTask'):
                return visitor.visitFugueDropnaTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueDropnaTask(self):
        localctx = fugue_sqlParser.FugueDropnaTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_fugueDropnaTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 543
            self.match(fugue_sqlParser.DROP)
            self.state = 544
            self.match(fugue_sqlParser.ROWS)
            self.state = 545
            self.match(fugue_sqlParser.IF)
            self.state = 546
            localctx.how = self._input.LT(1)
            _la = self._input.LA(1)
            if not (_la == 60 or _la == 65):
                localctx.how = self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 547
            _la = self._input.LA(1)
            if not (_la == 200 or _la == 201):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 550
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 21, self._ctx)
            if la_ == 1:
                self.state = 548
                self.match(fugue_sqlParser.ON)
                self.state = 549
                localctx.cols = self.fugueCols()
            self.state = 554
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 22, self._ctx)
            if la_ == 1:
                self.state = 552
                self.match(fugue_sqlParser.FROM)
                self.state = 553
                localctx.df = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueFillnaTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.params = None
            self.df = None

        def FILL(self):
            return self.getToken(fugue_sqlParser.FILL, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueFillnaTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueFillnaTask'):
                return visitor.visitFugueFillnaTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueFillnaTask(self):
        localctx = fugue_sqlParser.FugueFillnaTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_fugueFillnaTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 556
            self.match(fugue_sqlParser.FILL)
            self.state = 557
            _la = self._input.LA(1)
            if not (_la == 200 or _la == 201):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 558
            localctx.params = self.fugueParams()
            self.state = 561
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 23, self._ctx)
            if la_ == 1:
                self.state = 559
                self.match(fugue_sqlParser.FROM)
                self.state = 560
                localctx.df = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSampleTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.method = None
            self.seed = None
            self.df = None
            self.by = None

        def SAMPLE(self):
            return self.getToken(fugue_sqlParser.SAMPLE, 0)

        def fugueSampleMethod(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSampleMethodContext, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def SEED(self):
            return self.getToken(fugue_sqlParser.SEED, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def PREPARTITION(self):
            return self.getToken(fugue_sqlParser.PREPARTITION, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSampleTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSampleTask'):
                return visitor.visitFugueSampleTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueSampleTask(self):
        localctx = fugue_sqlParser.FugueSampleTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_fugueSampleTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 563
            self.match(fugue_sqlParser.SAMPLE)
            self.state = 565
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 238:
                self.state = 564
                self.match(fugue_sqlParser.REPLACE)
            self.state = 567
            localctx.method = self.fugueSampleMethod()
            self.state = 570
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 25, self._ctx)
            if la_ == 1:
                self.state = 568
                self.match(fugue_sqlParser.SEED)
                self.state = 569
                localctx.seed = self.match(fugue_sqlParser.INTEGER_VALUE)
            self.state = 574
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 26, self._ctx)
            if la_ == 1:
                self.state = 572
                self.match(fugue_sqlParser.FROM)
                self.state = 573
                localctx.df = self.fugueDataFrame()
            self.state = 579
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 27, self._ctx)
            if la_ == 1:
                self.state = 576
                self.match(fugue_sqlParser.PREPARTITION)
                self.state = 577
                self.match(fugue_sqlParser.BY)
                self.state = 578
                localctx.by = self.fugueCols()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueTakeTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.rows = None
            self.df = None
            self.partition = None
            self.presort = None
            self.na_position = None

        def TAKE(self):
            return self.getToken(fugue_sqlParser.TAKE, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def PRESORT(self):
            return self.getToken(fugue_sqlParser.PRESORT, 0)

        def FIRST(self):
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def LAST(self):
            return self.getToken(fugue_sqlParser.LAST, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueColsSort(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueTakeTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueTakeTask'):
                return visitor.visitFugueTakeTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueTakeTask(self):
        localctx = fugue_sqlParser.FugueTakeTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_fugueTakeTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 581
            self.match(fugue_sqlParser.TAKE)
            self.state = 584
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 28, self._ctx)
            if la_ == 1:
                self.state = 582
                localctx.rows = self.match(fugue_sqlParser.INTEGER_VALUE)
                self.state = 583
                _la = self._input.LA(1)
                if not (_la == 248 or _la == 249):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
            self.state = 588
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 29, self._ctx)
            if la_ == 1:
                self.state = 586
                self.match(fugue_sqlParser.FROM)
                self.state = 587
                localctx.df = self.fugueDataFrame()
            self.state = 593
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 30, self._ctx)
            if la_ == 1:
                self.state = 590
                localctx.partition = self.fuguePrepartition()
            elif la_ == 2:
                self.state = 591
                self.match(fugue_sqlParser.PRESORT)
                self.state = 592
                localctx.presort = self.fugueColsSort()
            self.state = 597
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 31, self._ctx)
            if la_ == 1:
                self.state = 595
                _la = self._input.LA(1)
                if not (_la == 200 or _la == 201):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 596
                localctx.na_position = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 140 or _la == 173):
                    localctx.na_position = self._errHandler.recoverInline(self)
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

    class FugueZipTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.dfs = None
            self.how = None
            self.by = None
            self.presort = None

        def ZIP(self):
            return self.getToken(fugue_sqlParser.ZIP, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def PRESORT(self):
            return self.getToken(fugue_sqlParser.PRESORT, 0)

        def fugueZipType(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueZipTypeContext, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def fugueColsSort(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueZipTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueZipTask'):
                return visitor.visitFugueZipTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueZipTask(self):
        localctx = fugue_sqlParser.FugueZipTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_fugueZipTask)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 599
            self.match(fugue_sqlParser.ZIP)
            self.state = 600
            localctx.dfs = self.fugueDataFrames()
            self.state = 602
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 32, self._ctx)
            if la_ == 1:
                self.state = 601
                localctx.how = self.fugueZipType()
            self.state = 606
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 33, self._ctx)
            if la_ == 1:
                self.state = 604
                self.match(fugue_sqlParser.BY)
                self.state = 605
                localctx.by = self.fugueCols()
            self.state = 610
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 34, self._ctx)
            if la_ == 1:
                self.state = 608
                self.match(fugue_sqlParser.PRESORT)
                self.state = 609
                localctx.presort = self.fugueColsSort()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueCreateTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.params = None

        def CREATE(self):
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def fugueSingleOutputExtensionCommon(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleOutputExtensionCommonContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueCreateTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCreateTask'):
                return visitor.visitFugueCreateTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueCreateTask(self):
        localctx = fugue_sqlParser.FugueCreateTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_fugueCreateTask)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 612
            self.match(fugue_sqlParser.CREATE)
            self.state = 613
            localctx.params = self.fugueSingleOutputExtensionCommon()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueCreateDataTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.data = None
            self.schema = None

        def CREATE(self):
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def fugueJsonArray(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonArrayContext, 0)

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def DATA(self):
            return self.getToken(fugue_sqlParser.DATA, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueCreateDataTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCreateDataTask'):
                return visitor.visitFugueCreateDataTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueCreateDataTask(self):
        localctx = fugue_sqlParser.FugueCreateDataTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_fugueCreateDataTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 615
            self.match(fugue_sqlParser.CREATE)
            self.state = 617
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 107:
                self.state = 616
                self.match(fugue_sqlParser.DATA)
            self.state = 619
            localctx.data = self.fugueJsonArray()
            self.state = 620
            self.match(fugue_sqlParser.SCHEMA)
            self.state = 621
            localctx.schema = self.fugueSchema()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueLoadTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.fmt = None
            self.path = None
            self.paths = None
            self.params = None
            self.columns = None

        def LOAD(self):
            return self.getToken(fugue_sqlParser.LOAD, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def fugueFileFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def fugueLoadColumns(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueLoadColumnsContext, 0)

        def fuguePath(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, 0)

        def fuguePaths(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePathsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueLoadTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueLoadTask'):
                return visitor.visitFugueLoadTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueLoadTask(self):
        localctx = fugue_sqlParser.FugueLoadTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_fugueLoadTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 623
            self.match(fugue_sqlParser.LOAD)
            self.state = 625
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 1924145348608 != 0:
                self.state = 624
                localctx.fmt = self.fugueFileFormat()
            self.state = 629
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [330]:
                self.state = 627
                localctx.path = self.fuguePath()
                pass
            elif token in [1]:
                self.state = 628
                localctx.paths = self.fuguePaths()
                pass
            else:
                raise NoViableAltException(self)
            self.state = 632
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 38, self._ctx)
            if la_ == 1:
                self.state = 631
                localctx.params = self.fugueParams()
            self.state = 636
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 39, self._ctx)
            if la_ == 1:
                self.state = 634
                self.match(fugue_sqlParser.COLUMNS)
                self.state = 635
                localctx.columns = self.fugueLoadColumns()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueOutputTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.dfs = None
            self.partition = None
            self.fugueUsing = None
            self.params = None

        def OUTPUT(self):
            return self.getToken(fugue_sqlParser.OUTPUT, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueOutputTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueOutputTask'):
                return visitor.visitFugueOutputTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueOutputTask(self):
        localctx = fugue_sqlParser.FugueOutputTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_fugueOutputTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 638
            self.match(fugue_sqlParser.OUTPUT)
            self.state = 640
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 40, self._ctx)
            if la_ == 1:
                self.state = 639
                localctx.dfs = self.fugueDataFrames()
            self.state = 643
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 642
                localctx.partition = self.fuguePrepartition()
            self.state = 645
            self.match(fugue_sqlParser.USING)
            self.state = 646
            localctx.fugueUsing = self.fugueExtension()
            self.state = 648
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                self.state = 647
                localctx.params = self.fugueParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FuguePrintTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.rows = None
            self.dfs = None
            self.count = None
            self.title = None

        def PRINT(self):
            return self.getToken(fugue_sqlParser.PRINT, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def TITLE(self):
            return self.getToken(fugue_sqlParser.TITLE, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def ROWCOUNT(self):
            return self.getToken(fugue_sqlParser.ROWCOUNT, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePrintTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePrintTask'):
                return visitor.visitFuguePrintTask(self)
            else:
                return visitor.visitChildren(self)

    def fuguePrintTask(self):
        localctx = fugue_sqlParser.FuguePrintTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_fuguePrintTask)
        self._la = 0
        try:
            self.state = 675
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 49, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 650
                self.match(fugue_sqlParser.PRINT)
                self.state = 651
                localctx.rows = self.match(fugue_sqlParser.INTEGER_VALUE)
                self.state = 652
                _la = self._input.LA(1)
                if not (_la == 248 or _la == 249):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 655
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 43, self._ctx)
                if la_ == 1:
                    self.state = 653
                    self.match(fugue_sqlParser.FROM)
                    self.state = 654
                    localctx.dfs = self.fugueDataFrames()
                self.state = 658
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 30:
                    self.state = 657
                    localctx.count = self.match(fugue_sqlParser.ROWCOUNT)
                self.state = 662
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 35:
                    self.state = 660
                    self.match(fugue_sqlParser.TITLE)
                    self.state = 661
                    localctx.title = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 664
                self.match(fugue_sqlParser.PRINT)
                self.state = 666
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 46, self._ctx)
                if la_ == 1:
                    self.state = 665
                    localctx.dfs = self.fugueDataFrames()
                self.state = 669
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 30:
                    self.state = 668
                    localctx.count = self.match(fugue_sqlParser.ROWCOUNT)
                self.state = 673
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 35:
                    self.state = 671
                    self.match(fugue_sqlParser.TITLE)
                    self.state = 672
                    localctx.title = self.match(fugue_sqlParser.STRING)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSaveTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.df = None
            self.partition = None
            self.m = None
            self.single = None
            self.fmt = None
            self.path = None
            self.params = None

        def SAVE(self):
            return self.getToken(fugue_sqlParser.SAVE, 0)

        def fugueSaveMode(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSaveModeContext, 0)

        def fuguePath(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueSingleFile(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

        def fugueFileFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSaveTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSaveTask'):
                return visitor.visitFugueSaveTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueSaveTask(self):
        localctx = fugue_sqlParser.FugueSaveTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_fugueSaveTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 677
            self.match(fugue_sqlParser.SAVE)
            self.state = 679
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 50, self._ctx)
            if la_ == 1:
                self.state = 678
                localctx.df = self.fugueDataFrame()
            self.state = 682
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 681
                localctx.partition = self.fuguePrepartition()
            self.state = 684
            localctx.m = self.fugueSaveMode()
            self.state = 686
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 41:
                self.state = 685
                localctx.single = self.fugueSingleFile()
            self.state = 689
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 1924145348608 != 0:
                self.state = 688
                localctx.fmt = self.fugueFileFormat()
            self.state = 691
            localctx.path = self.fuguePath()
            self.state = 693
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                self.state = 692
                localctx.params = self.fugueParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueOutputTransformTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.dfs = None
            self.partition = None
            self.fugueUsing = None
            self.params = None
            self.callback = None

        def OUTTRANSFORM(self):
            return self.getToken(fugue_sqlParser.OUTTRANSFORM, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def fugueExtension(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueExtensionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, i)

        def CALLBACK(self):
            return self.getToken(fugue_sqlParser.CALLBACK, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueOutputTransformTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueOutputTransformTask'):
                return visitor.visitFugueOutputTransformTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueOutputTransformTask(self):
        localctx = fugue_sqlParser.FugueOutputTransformTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_fugueOutputTransformTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 695
            self.match(fugue_sqlParser.OUTTRANSFORM)
            self.state = 697
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 55, self._ctx)
            if la_ == 1:
                self.state = 696
                localctx.dfs = self.fugueDataFrames()
            self.state = 700
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                self.state = 699
                localctx.partition = self.fuguePrepartition()
            self.state = 702
            self.match(fugue_sqlParser.USING)
            self.state = 703
            localctx.fugueUsing = self.fugueExtension()
            self.state = 705
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                self.state = 704
                localctx.params = self.fugueParams()
            self.state = 709
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 55:
                self.state = 707
                self.match(fugue_sqlParser.CALLBACK)
                self.state = 708
                localctx.callback = self.fugueExtension()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueModuleTaskContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.assign = None
            self.dfs = None
            self.fugueUsing = None
            self.params = None

        def SUB(self):
            return self.getToken(fugue_sqlParser.SUB, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def fugueAssignment(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentContext, 0)

        def fugueDataFrames(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramesContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueModuleTask

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueModuleTask'):
                return visitor.visitFugueModuleTask(self)
            else:
                return visitor.visitChildren(self)

    def fugueModuleTask(self):
        localctx = fugue_sqlParser.FugueModuleTaskContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_fugueModuleTask)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 712
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la - 58 & ~63 == 0 and 1 << _la - 58 & -1 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -1 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -1 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1152921504606846975 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0):
                self.state = 711
                localctx.assign = self.fugueAssignment()
            self.state = 714
            self.match(fugue_sqlParser.SUB)
            self.state = 716
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 60, self._ctx)
            if la_ == 1:
                self.state = 715
                localctx.dfs = self.fugueDataFrames()
            self.state = 718
            self.match(fugue_sqlParser.USING)
            self.state = 719
            localctx.fugueUsing = self.fugueExtension()
            self.state = 721
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                self.state = 720
                localctx.params = self.fugueParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSqlEngineContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.fugueUsing = None
            self.params = None

        def CONNECT(self):
            return self.getToken(fugue_sqlParser.CONNECT, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSqlEngine

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSqlEngine'):
                return visitor.visitFugueSqlEngine(self)
            else:
                return visitor.visitChildren(self)

    def fugueSqlEngine(self):
        localctx = fugue_sqlParser.FugueSqlEngineContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_fugueSqlEngine)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 723
            self.match(fugue_sqlParser.CONNECT)
            self.state = 724
            localctx.fugueUsing = self.fugueExtension()
            self.state = 726
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                self.state = 725
                localctx.params = self.fugueParams()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSingleFileContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.single = None

        def SINGLE(self):
            return self.getToken(fugue_sqlParser.SINGLE, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSingleFile

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSingleFile'):
                return visitor.visitFugueSingleFile(self)
            else:
                return visitor.visitChildren(self)

    def fugueSingleFile(self):
        localctx = fugue_sqlParser.FugueSingleFileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_fugueSingleFile)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 728
            localctx.single = self.match(fugue_sqlParser.SINGLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueLoadColumnsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.schema = None
            self.cols = None

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueLoadColumns

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueLoadColumns'):
                return visitor.visitFugueLoadColumns(self)
            else:
                return visitor.visitChildren(self)

    def fugueLoadColumns(self):
        localctx = fugue_sqlParser.FugueLoadColumnsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_fugueLoadColumns)
        try:
            self.state = 732
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 63, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 730
                localctx.schema = self.fugueSchema()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 731
                localctx.cols = self.fugueCols()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSaveModeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def APPEND(self):
            return self.getToken(fugue_sqlParser.APPEND, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSaveMode

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSaveMode'):
                return visitor.visitFugueSaveMode(self)
            else:
                return visitor.visitChildren(self)

    def fugueSaveMode(self):
        localctx = fugue_sqlParser.FugueSaveModeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_fugueSaveMode)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 734
            _la = self._input.LA(1)
            if not (_la == 37 or _la == 215 or _la == 280):
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

    class FugueFileFormatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PARQUET(self):
            return self.getToken(fugue_sqlParser.PARQUET, 0)

        def CSV(self):
            return self.getToken(fugue_sqlParser.CSV, 0)

        def JSON(self):
            return self.getToken(fugue_sqlParser.JSON, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueFileFormat

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueFileFormat'):
                return visitor.visitFugueFileFormat(self)
            else:
                return visitor.visitChildren(self)

    def fugueFileFormat(self):
        localctx = fugue_sqlParser.FugueFileFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_fugueFileFormat)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 736
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & 1924145348608 != 0):
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

    class FuguePathContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePath

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePath'):
                return visitor.visitFuguePath(self)
            else:
                return visitor.visitChildren(self)

    def fuguePath(self):
        localctx = fugue_sqlParser.FuguePathContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_fuguePath)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 738
            self.match(fugue_sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FuguePathsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fuguePath(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FuguePathContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePaths

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePaths'):
                return visitor.visitFuguePaths(self)
            else:
                return visitor.visitChildren(self)

    def fuguePaths(self):
        localctx = fugue_sqlParser.FuguePathsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_fuguePaths)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 740
            self.match(fugue_sqlParser.T__0)
            self.state = 741
            self.fuguePath()
            self.state = 746
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 742
                self.match(fugue_sqlParser.T__1)
                self.state = 743
                self.fuguePath()
                self.state = 748
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 749
            self.match(fugue_sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueCheckpointContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueCheckpoint

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class FugueCheckpointDeterministicContext(FugueCheckpointContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.ns = None
            self.partition = None
            self.single = None
            self.params = None
            self.copyFrom(ctx)

        def DETERMINISTIC(self):
            return self.getToken(fugue_sqlParser.DETERMINISTIC, 0)

        def CHECKPOINT(self):
            return self.getToken(fugue_sqlParser.CHECKPOINT, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def fugueCheckpointNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueCheckpointNamespaceContext, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueSingleFile(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCheckpointDeterministic'):
                return visitor.visitFugueCheckpointDeterministic(self)
            else:
                return visitor.visitChildren(self)

    class FugueCheckpointWeakContext(FugueCheckpointContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.params = None
            self.copyFrom(ctx)

        def PERSIST(self):
            return self.getToken(fugue_sqlParser.PERSIST, 0)

        def WEAK(self):
            return self.getToken(fugue_sqlParser.WEAK, 0)

        def CHECKPOINT(self):
            return self.getToken(fugue_sqlParser.CHECKPOINT, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCheckpointWeak'):
                return visitor.visitFugueCheckpointWeak(self)
            else:
                return visitor.visitChildren(self)

    class FugueCheckpointStrongContext(FugueCheckpointContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.partition = None
            self.single = None
            self.params = None
            self.copyFrom(ctx)

        def CHECKPOINT(self):
            return self.getToken(fugue_sqlParser.CHECKPOINT, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def STRONG(self):
            return self.getToken(fugue_sqlParser.STRONG, 0)

        def fuguePrepartition(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePrepartitionContext, 0)

        def fugueSingleFile(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSingleFileContext, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCheckpointStrong'):
                return visitor.visitFugueCheckpointStrong(self)
            else:
                return visitor.visitChildren(self)

    def fugueCheckpoint(self):
        localctx = fugue_sqlParser.FugueCheckpointContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_fugueCheckpoint)
        self._la = 0
        try:
            self.state = 795
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 78, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.FugueCheckpointWeakContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 752
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 175:
                    self.state = 751
                    self.match(fugue_sqlParser.LAZY)
                self.state = 757
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [24]:
                    self.state = 754
                    self.match(fugue_sqlParser.PERSIST)
                    pass
                elif token in [43]:
                    self.state = 755
                    self.match(fugue_sqlParser.WEAK)
                    self.state = 756
                    self.match(fugue_sqlParser.CHECKPOINT)
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 760
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                    self.state = 759
                    localctx.params = self.fugueParams()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.FugueCheckpointStrongContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 763
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 175:
                    self.state = 762
                    self.match(fugue_sqlParser.LAZY)
                self.state = 766
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 44:
                    self.state = 765
                    self.match(fugue_sqlParser.STRONG)
                self.state = 768
                self.match(fugue_sqlParser.CHECKPOINT)
                self.state = 770
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                    self.state = 769
                    localctx.partition = self.fuguePrepartition()
                self.state = 773
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 41:
                    self.state = 772
                    localctx.single = self.fugueSingleFile()
                self.state = 776
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                    self.state = 775
                    localctx.params = self.fugueParams()
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.FugueCheckpointDeterministicContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 779
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 175:
                    self.state = 778
                    self.match(fugue_sqlParser.LAZY)
                self.state = 781
                self.match(fugue_sqlParser.DETERMINISTIC)
                self.state = 782
                self.match(fugue_sqlParser.CHECKPOINT)
                self.state = 784
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 330:
                    self.state = 783
                    localctx.ns = self.fugueCheckpointNamespace()
                self.state = 787
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 4302831616 != 0:
                    self.state = 786
                    localctx.partition = self.fuguePrepartition()
                self.state = 790
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 41:
                    self.state = 789
                    localctx.single = self.fugueSingleFile()
                self.state = 793
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 67109152 != 0:
                    self.state = 792
                    localctx.params = self.fugueParams()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueCheckpointNamespaceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueCheckpointNamespace

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCheckpointNamespace'):
                return visitor.visitFugueCheckpointNamespace(self)
            else:
                return visitor.visitChildren(self)

    def fugueCheckpointNamespace(self):
        localctx = fugue_sqlParser.FugueCheckpointNamespaceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 64, self.RULE_fugueCheckpointNamespace)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 797
            self.match(fugue_sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueYieldContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None

        def YIELD(self):
            return self.getToken(fugue_sqlParser.YIELD, 0)

        def FILE(self):
            return self.getToken(fugue_sqlParser.FILE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def DATAFRAME(self):
            return self.getToken(fugue_sqlParser.DATAFRAME, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueYield

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueYield'):
                return visitor.visitFugueYield(self)
            else:
                return visitor.visitChildren(self)

    def fugueYield(self):
        localctx = fugue_sqlParser.FugueYieldContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_fugueYield)
        self._la = 0
        try:
            self.state = 814
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 82, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 799
                self.match(fugue_sqlParser.YIELD)
                self.state = 800
                _la = self._input.LA(1)
                if not (_la == 57 or _la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 803
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 79, self._ctx)
                if la_ == 1:
                    self.state = 801
                    self.match(fugue_sqlParser.AS)
                    self.state = 802
                    localctx.name = self.fugueIdentifier()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 805
                self.match(fugue_sqlParser.YIELD)
                self.state = 807
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 183:
                    self.state = 806
                    self.match(fugue_sqlParser.LOCAL)
                self.state = 809
                self.match(fugue_sqlParser.DATAFRAME)
                self.state = 812
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 81, self._ctx)
                if la_ == 1:
                    self.state = 810
                    self.match(fugue_sqlParser.AS)
                    self.state = 811
                    localctx.name = self.fugueIdentifier()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueBroadcastContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BROADCAST(self):
            return self.getToken(fugue_sqlParser.BROADCAST, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueBroadcast

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueBroadcast'):
                return visitor.visitFugueBroadcast(self)
            else:
                return visitor.visitChildren(self)

    def fugueBroadcast(self):
        localctx = fugue_sqlParser.FugueBroadcastContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_fugueBroadcast)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 816
            self.match(fugue_sqlParser.BROADCAST)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueDataFramesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDataFrames

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class FugueDataFramesDictContext(FugueDataFramesContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueDataFramePair(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueDataFramePairContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramePairContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFramesDict'):
                return visitor.visitFugueDataFramesDict(self)
            else:
                return visitor.visitChildren(self)

    class FugueDataFramesListContext(FugueDataFramesContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueDataFrame(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueDataFrameContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFramesList'):
                return visitor.visitFugueDataFramesList(self)
            else:
                return visitor.visitChildren(self)

    def fugueDataFrames(self):
        localctx = fugue_sqlParser.FugueDataFramesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 70, self.RULE_fugueDataFrames)
        try:
            self.state = 834
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 85, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.FugueDataFramesListContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 818
                self.fugueDataFrame()
                self.state = 823
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 83, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 819
                        self.match(fugue_sqlParser.T__1)
                        self.state = 820
                        self.fugueDataFrame()
                    self.state = 825
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 83, self._ctx)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.FugueDataFramesDictContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 826
                self.fugueDataFramePair()
                self.state = 831
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 84, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 827
                        self.match(fugue_sqlParser.T__1)
                        self.state = 828
                        self.fugueDataFramePair()
                    self.state = 833
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 84, self._ctx)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueDataFramePairContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def fugueDataFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDataFramePair

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFramePair'):
                return visitor.visitFugueDataFramePair(self)
            else:
                return visitor.visitChildren(self)

    def fugueDataFramePair(self):
        localctx = fugue_sqlParser.FugueDataFramePairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 72, self.RULE_fugueDataFramePair)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 836
            localctx.key = self.fugueIdentifier()
            self.state = 837
            _la = self._input.LA(1)
            if not (_la == 4 or _la == 310):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 838
            localctx.value = self.fugueDataFrame()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueDataFrameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDataFrame

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class FugueDataFrameSourceContext(FugueDataFrameContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def fugueDataFrameMember(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameMemberContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFrameSource'):
                return visitor.visitFugueDataFrameSource(self)
            else:
                return visitor.visitChildren(self)

    class FugueDataFrameNestedContext(FugueDataFrameContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.task = None
            self.copyFrom(ctx)

        def fugueNestableTask(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFrameNested'):
                return visitor.visitFugueDataFrameNested(self)
            else:
                return visitor.visitChildren(self)

    def fugueDataFrame(self):
        localctx = fugue_sqlParser.FugueDataFrameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 74, self.RULE_fugueDataFrame)
        try:
            self.state = 848
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                localctx = fugue_sqlParser.FugueDataFrameSourceContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 840
                self.fugueIdentifier()
                self.state = 842
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 86, self._ctx)
                if la_ == 1:
                    self.state = 841
                    self.fugueDataFrameMember()
                pass
            elif token in [5]:
                localctx = fugue_sqlParser.FugueDataFrameNestedContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 844
                self.match(fugue_sqlParser.T__4)
                self.state = 845
                localctx.task = self.fugueNestableTask()
                self.state = 846
                self.match(fugue_sqlParser.T__5)
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

    class FugueDataFrameMemberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.index = None
            self.key = None

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueDataFrameMember

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueDataFrameMember'):
                return visitor.visitFugueDataFrameMember(self)
            else:
                return visitor.visitChildren(self)

    def fugueDataFrameMember(self):
        localctx = fugue_sqlParser.FugueDataFrameMemberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 76, self.RULE_fugueDataFrameMember)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 850
            self.match(fugue_sqlParser.T__0)
            self.state = 853
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [334]:
                self.state = 851
                localctx.index = self.match(fugue_sqlParser.INTEGER_VALUE)
                pass
            elif token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.state = 852
                localctx.key = self.fugueIdentifier()
                pass
            else:
                raise NoViableAltException(self)
            self.state = 855
            self.match(fugue_sqlParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueAssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.varname = None
            self.sign = None

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def fugueAssignmentSign(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueAssignmentSignContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueAssignment

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueAssignment'):
                return visitor.visitFugueAssignment(self)
            else:
                return visitor.visitChildren(self)

    def fugueAssignment(self):
        localctx = fugue_sqlParser.FugueAssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 78, self.RULE_fugueAssignment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 857
            localctx.varname = self.fugueIdentifier()
            self.state = 858
            localctx.sign = self.fugueAssignmentSign()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueAssignmentSignContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueAssignmentSign

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueAssignmentSign'):
                return visitor.visitFugueAssignmentSign(self)
            else:
                return visitor.visitChildren(self)

    def fugueAssignmentSign(self):
        localctx = fugue_sqlParser.FugueAssignmentSignContext(self, self._ctx, self.state)
        self.enterRule(localctx, 80, self.RULE_fugueAssignmentSign)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 860
            self.match(fugue_sqlParser.EQUAL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSingleOutputExtensionCommonWildContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.fugueUsing = None
            self.params = None
            self.schema = None

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def fugueWildSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueWildSchemaContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSingleOutputExtensionCommonWild

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSingleOutputExtensionCommonWild'):
                return visitor.visitFugueSingleOutputExtensionCommonWild(self)
            else:
                return visitor.visitChildren(self)

    def fugueSingleOutputExtensionCommonWild(self):
        localctx = fugue_sqlParser.FugueSingleOutputExtensionCommonWildContext(self, self._ctx, self.state)
        self.enterRule(localctx, 82, self.RULE_fugueSingleOutputExtensionCommonWild)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 862
            self.match(fugue_sqlParser.USING)
            self.state = 863
            localctx.fugueUsing = self.fugueExtension()
            self.state = 865
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 89, self._ctx)
            if la_ == 1:
                self.state = 864
                localctx.params = self.fugueParams()
            self.state = 869
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 90, self._ctx)
            if la_ == 1:
                self.state = 867
                self.match(fugue_sqlParser.SCHEMA)
                self.state = 868
                localctx.schema = self.fugueWildSchema()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSingleOutputExtensionCommonContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.fugueUsing = None
            self.params = None
            self.schema = None

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def fugueExtension(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def fugueParams(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSingleOutputExtensionCommon

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSingleOutputExtensionCommon'):
                return visitor.visitFugueSingleOutputExtensionCommon(self)
            else:
                return visitor.visitChildren(self)

    def fugueSingleOutputExtensionCommon(self):
        localctx = fugue_sqlParser.FugueSingleOutputExtensionCommonContext(self, self._ctx, self.state)
        self.enterRule(localctx, 84, self.RULE_fugueSingleOutputExtensionCommon)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 871
            self.match(fugue_sqlParser.USING)
            self.state = 872
            localctx.fugueUsing = self.fugueExtension()
            self.state = 874
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 91, self._ctx)
            if la_ == 1:
                self.state = 873
                localctx.params = self.fugueParams()
            self.state = 878
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 92, self._ctx)
            if la_ == 1:
                self.state = 876
                self.match(fugue_sqlParser.SCHEMA)
                self.state = 877
                localctx.schema = self.fugueSchema()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueExtensionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.domain = None

        def fugueIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueExtension

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueExtension'):
                return visitor.visitFugueExtension(self)
            else:
                return visitor.visitChildren(self)

    def fugueExtension(self):
        localctx = fugue_sqlParser.FugueExtensionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 86, self.RULE_fugueExtension)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 883
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 93, self._ctx)
            if la_ == 1:
                self.state = 880
                localctx.domain = self.fugueIdentifier()
                self.state = 881
                self.match(fugue_sqlParser.T__3)
            self.state = 885
            self.fugueIdentifier()
            self.state = 890
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 94, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 886
                    self.match(fugue_sqlParser.T__6)
                    self.state = 887
                    self.fugueIdentifier()
                self.state = 892
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 94, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSampleMethodContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.percentage = None
            self.rows = None

        def PERCENTLIT(self):
            return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

        def PERCENT(self):
            return self.getToken(fugue_sqlParser.PERCENT, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def APPROX(self):
            return self.getToken(fugue_sqlParser.APPROX, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSampleMethod

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSampleMethod'):
                return visitor.visitFugueSampleMethod(self)
            else:
                return visitor.visitChildren(self)

    def fugueSampleMethod(self):
        localctx = fugue_sqlParser.FugueSampleMethodContext(self, self._ctx, self.state)
        self.enterRule(localctx, 88, self.RULE_fugueSampleMethod)
        self._la = 0
        try:
            self.state = 900
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 96, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 893
                localctx.percentage = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 334 or _la == 336):
                    localctx.percentage = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 894
                _la = self._input.LA(1)
                if not (_la == 219 or _la == 323):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 896
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 50:
                    self.state = 895
                    self.match(fugue_sqlParser.APPROX)
                self.state = 898
                localctx.rows = self.match(fugue_sqlParser.INTEGER_VALUE)
                self.state = 899
                self.match(fugue_sqlParser.ROWS)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueZipTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CROSS(self):
            return self.getToken(fugue_sqlParser.CROSS, 0)

        def INNER(self):
            return self.getToken(fugue_sqlParser.INNER, 0)

        def LEFT(self):
            return self.getToken(fugue_sqlParser.LEFT, 0)

        def OUTER(self):
            return self.getToken(fugue_sqlParser.OUTER, 0)

        def RIGHT(self):
            return self.getToken(fugue_sqlParser.RIGHT, 0)

        def FULL(self):
            return self.getToken(fugue_sqlParser.FULL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueZipType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueZipType'):
                return visitor.visitFugueZipType(self)
            else:
                return visitor.visitChildren(self)

    def fugueZipType(self):
        localctx = fugue_sqlParser.FugueZipTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 90, self.RULE_fugueZipType)
        try:
            self.state = 910
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [100]:
                self.enterOuterAlt(localctx, 1)
                self.state = 902
                self.match(fugue_sqlParser.CROSS)
                pass
            elif token in [162]:
                self.enterOuterAlt(localctx, 2)
                self.state = 903
                self.match(fugue_sqlParser.INNER)
                pass
            elif token in [177]:
                self.enterOuterAlt(localctx, 3)
                self.state = 904
                self.match(fugue_sqlParser.LEFT)
                self.state = 905
                self.match(fugue_sqlParser.OUTER)
                pass
            elif token in [242]:
                self.enterOuterAlt(localctx, 4)
                self.state = 906
                self.match(fugue_sqlParser.RIGHT)
                self.state = 907
                self.match(fugue_sqlParser.OUTER)
                pass
            elif token in [147]:
                self.enterOuterAlt(localctx, 5)
                self.state = 908
                self.match(fugue_sqlParser.FULL)
                self.state = 909
                self.match(fugue_sqlParser.OUTER)
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

    class FuguePrepartitionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.algo = None
            self.num = None
            self.by = None
            self.presort = None

        def PREPARTITION(self):
            return self.getToken(fugue_sqlParser.PREPARTITION, 0)

        def fuguePartitionNum(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumContext, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def PRESORT(self):
            return self.getToken(fugue_sqlParser.PRESORT, 0)

        def fuguePartitionAlgo(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionAlgoContext, 0)

        def fugueCols(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsContext, 0)

        def fugueColsSort(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColsSortContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePrepartition

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePrepartition'):
                return visitor.visitFuguePrepartition(self)
            else:
                return visitor.visitChildren(self)

    def fuguePrepartition(self):
        localctx = fugue_sqlParser.FuguePrepartitionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 92, self.RULE_fuguePrepartition)
        self._la = 0
        try:
            self.state = 935
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 103, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 913
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 7864320 != 0:
                    self.state = 912
                    localctx.algo = self.fuguePartitionAlgo()
                self.state = 915
                self.match(fugue_sqlParser.PREPARTITION)
                self.state = 916
                localctx.num = self.fuguePartitionNum(0)
                self.state = 919
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 99, self._ctx)
                if la_ == 1:
                    self.state = 917
                    self.match(fugue_sqlParser.BY)
                    self.state = 918
                    localctx.by = self.fugueCols()
                self.state = 923
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 100, self._ctx)
                if la_ == 1:
                    self.state = 921
                    self.match(fugue_sqlParser.PRESORT)
                    self.state = 922
                    localctx.presort = self.fugueColsSort()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 926
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & 7864320 != 0:
                    self.state = 925
                    localctx.algo = self.fuguePartitionAlgo()
                self.state = 928
                self.match(fugue_sqlParser.PREPARTITION)
                self.state = 929
                self.match(fugue_sqlParser.BY)
                self.state = 930
                localctx.by = self.fugueCols()
                self.state = 933
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 102, self._ctx)
                if la_ == 1:
                    self.state = 931
                    self.match(fugue_sqlParser.PRESORT)
                    self.state = 932
                    localctx.presort = self.fugueColsSort()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FuguePartitionAlgoContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def HASH(self):
            return self.getToken(fugue_sqlParser.HASH, 0)

        def RAND(self):
            return self.getToken(fugue_sqlParser.RAND, 0)

        def EVEN(self):
            return self.getToken(fugue_sqlParser.EVEN, 0)

        def COARSE(self):
            return self.getToken(fugue_sqlParser.COARSE, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePartitionAlgo

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePartitionAlgo'):
                return visitor.visitFuguePartitionAlgo(self)
            else:
                return visitor.visitChildren(self)

    def fuguePartitionAlgo(self):
        localctx = fugue_sqlParser.FuguePartitionAlgoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 94, self.RULE_fuguePartitionAlgo)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 937
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & 7864320 != 0):
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

    class FuguePartitionNumContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fuguePartitionNumber(self):
            return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumberContext, 0)

        def fuguePartitionNum(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FuguePartitionNumContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FuguePartitionNumContext, i)

        def PLUS(self):
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def SLASH(self):
            return self.getToken(fugue_sqlParser.SLASH, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePartitionNum

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePartitionNum'):
                return visitor.visitFuguePartitionNum(self)
            else:
                return visitor.visitChildren(self)

    def fuguePartitionNum(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = fugue_sqlParser.FuguePartitionNumContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 96
        self.enterRecursionRule(localctx, 96, self.RULE_fuguePartitionNum, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 945
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [30, 31, 320, 334, 336]:
                self.state = 940
                self.fuguePartitionNumber()
                pass
            elif token in [5]:
                self.state = 941
                self.match(fugue_sqlParser.T__4)
                self.state = 942
                self.fuguePartitionNum(0)
                self.state = 943
                self.match(fugue_sqlParser.T__5)
                pass
            else:
                raise NoViableAltException(self)
            self._ctx.stop = self._input.LT(-1)
            self.state = 952
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 105, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = fugue_sqlParser.FuguePartitionNumContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_fuguePartitionNum)
                    self.state = 947
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                    self.state = 948
                    _la = self._input.LA(1)
                    if not (_la - 319 & ~63 == 0 and 1 << _la - 319 & 15 != 0):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 949
                    self.fuguePartitionNum(2)
                self.state = 954
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 105, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class FuguePartitionNumberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def ROWCOUNT(self):
            return self.getToken(fugue_sqlParser.ROWCOUNT, 0)

        def CONCURRENCY(self):
            return self.getToken(fugue_sqlParser.CONCURRENCY, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fuguePartitionNumber

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFuguePartitionNumber'):
                return visitor.visitFuguePartitionNumber(self)
            else:
                return visitor.visitChildren(self)

    def fuguePartitionNumber(self):
        localctx = fugue_sqlParser.FuguePartitionNumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 98, self.RULE_fuguePartitionNumber)
        self._la = 0
        try:
            self.state = 965
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 108, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 956
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 955
                    self.match(fugue_sqlParser.MINUS)
                self.state = 958
                self.match(fugue_sqlParser.DECIMAL_VALUE)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 960
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 959
                    self.match(fugue_sqlParser.MINUS)
                self.state = 962
                self.match(fugue_sqlParser.INTEGER_VALUE)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 963
                self.match(fugue_sqlParser.ROWCOUNT)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 964
                self.match(fugue_sqlParser.CONCURRENCY)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueParamsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueParams

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class FugueParamsPairsContext(FugueParamsContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.pairs = None
            self.copyFrom(ctx)

        def PARAMS(self):
            return self.getToken(fugue_sqlParser.PARAMS, 0)

        def fugueJsonPairs(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairsContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueParamsPairs'):
                return visitor.visitFugueParamsPairs(self)
            else:
                return visitor.visitChildren(self)

    class FugueParamsObjContext(FugueParamsContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.obj = None
            self.copyFrom(ctx)

        def fugueJsonObj(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonObjContext, 0)

        def PARAMS(self):
            return self.getToken(fugue_sqlParser.PARAMS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueParamsObj'):
                return visitor.visitFugueParamsObj(self)
            else:
                return visitor.visitChildren(self)

    def fugueParams(self):
        localctx = fugue_sqlParser.FugueParamsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 100, self.RULE_fugueParams)
        self._la = 0
        try:
            self.state = 973
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 110, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.FugueParamsPairsContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 967
                self.match(fugue_sqlParser.PARAMS)
                self.state = 968
                localctx.pairs = self.fugueJsonPairs()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.FugueParamsObjContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 970
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 26:
                    self.state = 969
                    self.match(fugue_sqlParser.PARAMS)
                self.state = 972
                localctx.obj = self.fugueJsonObj()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueColsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueColumnIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueColumnIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueColumnIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueCols

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueCols'):
                return visitor.visitFugueCols(self)
            else:
                return visitor.visitChildren(self)

    def fugueCols(self):
        localctx = fugue_sqlParser.FugueColsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 102, self.RULE_fugueCols)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 975
            self.fugueColumnIdentifier()
            self.state = 980
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 111, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 976
                    self.match(fugue_sqlParser.T__1)
                    self.state = 977
                    self.fugueColumnIdentifier()
                self.state = 982
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 111, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueColsSortContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueColSort(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueColSortContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueColSortContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueColsSort

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueColsSort'):
                return visitor.visitFugueColsSort(self)
            else:
                return visitor.visitChildren(self)

    def fugueColsSort(self):
        localctx = fugue_sqlParser.FugueColsSortContext(self, self._ctx, self.state)
        self.enterRule(localctx, 104, self.RULE_fugueColsSort)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 983
            self.fugueColSort()
            self.state = 988
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 112, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 984
                    self.match(fugue_sqlParser.T__1)
                    self.state = 985
                    self.fugueColSort()
                self.state = 990
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 112, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueColSortContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueColumnIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueColumnIdentifierContext, 0)

        def ASC(self):
            return self.getToken(fugue_sqlParser.ASC, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueColSort

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueColSort'):
                return visitor.visitFugueColSort(self)
            else:
                return visitor.visitChildren(self)

    def fugueColSort(self):
        localctx = fugue_sqlParser.FugueColSortContext(self, self._ctx, self.state)
        self.enterRule(localctx, 106, self.RULE_fugueColSort)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 991
            self.fugueColumnIdentifier()
            self.state = 993
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 113, self._ctx)
            if la_ == 1:
                self.state = 992
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 115):
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

    class FugueColumnIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueColumnIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueColumnIdentifier'):
                return visitor.visitFugueColumnIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def fugueColumnIdentifier(self):
        localctx = fugue_sqlParser.FugueColumnIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 108, self.RULE_fugueColumnIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 995
            self.fugueIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueRenameExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueRenamePair(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueRenamePairContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueRenamePairContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueRenameExpression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueRenameExpression'):
                return visitor.visitFugueRenameExpression(self)
            else:
                return visitor.visitChildren(self)

    def fugueRenameExpression(self):
        localctx = fugue_sqlParser.FugueRenameExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 110, self.RULE_fugueRenameExpression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 997
            self.fugueRenamePair()
            self.state = 1002
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 114, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 998
                    self.match(fugue_sqlParser.T__1)
                    self.state = 999
                    self.fugueRenamePair()
                self.state = 1004
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 114, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueWildSchemaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueWildSchemaPair(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueWildSchemaPairContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueWildSchemaPairContext, i)

        def fugueSchemaOp(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaOpContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaOpContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueWildSchema

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueWildSchema'):
                return visitor.visitFugueWildSchema(self)
            else:
                return visitor.visitChildren(self)

    def fugueWildSchema(self):
        localctx = fugue_sqlParser.FugueWildSchemaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 112, self.RULE_fugueWildSchema)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1005
            self.fugueWildSchemaPair()
            self.state = 1010
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 115, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1006
                    self.match(fugue_sqlParser.T__1)
                    self.state = 1007
                    self.fugueWildSchemaPair()
                self.state = 1012
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 115, self._ctx)
            self.state = 1016
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 116, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1013
                    self.fugueSchemaOp()
                self.state = 1018
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 116, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueWildSchemaPairContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.pair = None

        def fugueSchemaPair(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaPairContext, 0)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueWildSchemaPair

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueWildSchemaPair'):
                return visitor.visitFugueWildSchemaPair(self)
            else:
                return visitor.visitChildren(self)

    def fugueWildSchemaPair(self):
        localctx = fugue_sqlParser.FugueWildSchemaPairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 114, self.RULE_fugueWildSchemaPair)
        try:
            self.state = 1021
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1019
                localctx.pair = self.fugueSchemaPair()
                pass
            elif token in [321]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1020
                self.match(fugue_sqlParser.ASTERISK)
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

    class FugueSchemaOpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueSchemaKey(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaKeyContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, i)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def TILDE(self):
            return self.getToken(fugue_sqlParser.TILDE, 0)

        def PLUS(self):
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSchemaOp

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaOp'):
                return visitor.visitFugueSchemaOp(self)
            else:
                return visitor.visitChildren(self)

    def fugueSchemaOp(self):
        localctx = fugue_sqlParser.FugueSchemaOpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 116, self.RULE_fugueSchemaOp)
        self._la = 0
        try:
            self.state = 1034
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [320, 325]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1023
                _la = self._input.LA(1)
                if not (_la == 320 or _la == 325):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1024
                self.fugueSchemaKey()
                self.state = 1029
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 118, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1025
                        self.match(fugue_sqlParser.T__1)
                        self.state = 1026
                        self.fugueSchemaKey()
                    self.state = 1031
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 118, self._ctx)
                pass
            elif token in [319]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1032
                self.match(fugue_sqlParser.PLUS)
                self.state = 1033
                self.fugueSchema()
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

    class FugueSchemaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueSchemaPair(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaPairContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaPairContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSchema

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchema'):
                return visitor.visitFugueSchema(self)
            else:
                return visitor.visitChildren(self)

    def fugueSchema(self):
        localctx = fugue_sqlParser.FugueSchemaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 118, self.RULE_fugueSchema)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1036
            self.fugueSchemaPair()
            self.state = 1041
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 120, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1037
                    self.match(fugue_sqlParser.T__1)
                    self.state = 1038
                    self.fugueSchemaPair()
                self.state = 1043
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 120, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSchemaPairContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def fugueSchemaKey(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, 0)

        def fugueSchemaType(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSchemaPair

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaPair'):
                return visitor.visitFugueSchemaPair(self)
            else:
                return visitor.visitChildren(self)

    def fugueSchemaPair(self):
        localctx = fugue_sqlParser.FugueSchemaPairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 120, self.RULE_fugueSchemaPair)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1044
            localctx.key = self.fugueSchemaKey()
            self.state = 1045
            self.match(fugue_sqlParser.T__3)
            self.state = 1046
            localctx.value = self.fugueSchemaType()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSchemaKeyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSchemaKey

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaKey'):
                return visitor.visitFugueSchemaKey(self)
            else:
                return visitor.visitChildren(self)

    def fugueSchemaKey(self):
        localctx = fugue_sqlParser.FugueSchemaKeyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 122, self.RULE_fugueSchemaKey)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1048
            self.fugueIdentifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueSchemaTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueSchemaType

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class FugueSchemaSimpleTypeContext(FugueSchemaTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaSimpleType'):
                return visitor.visitFugueSchemaSimpleType(self)
            else:
                return visitor.visitChildren(self)

    class FugueSchemaMapTypeContext(FugueSchemaTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def LT(self):
            return self.getToken(fugue_sqlParser.LT, 0)

        def fugueSchemaType(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaTypeContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, i)

        def GT(self):
            return self.getToken(fugue_sqlParser.GT, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaMapType'):
                return visitor.visitFugueSchemaMapType(self)
            else:
                return visitor.visitChildren(self)

    class FugueSchemaStructTypeContext(FugueSchemaTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueSchema(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaStructType'):
                return visitor.visitFugueSchemaStructType(self)
            else:
                return visitor.visitChildren(self)

    class FugueSchemaListTypeContext(FugueSchemaTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueSchemaType(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueSchemaListType'):
                return visitor.visitFugueSchemaListType(self)
            else:
                return visitor.visitChildren(self)

    def fugueSchemaType(self):
        localctx = fugue_sqlParser.FugueSchemaTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 124, self.RULE_fugueSchemaType)
        try:
            self.state = 1065
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                localctx = fugue_sqlParser.FugueSchemaSimpleTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1050
                self.fugueIdentifier()
                pass
            elif token in [1]:
                localctx = fugue_sqlParser.FugueSchemaListTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1051
                self.match(fugue_sqlParser.T__0)
                self.state = 1052
                self.fugueSchemaType()
                self.state = 1053
                self.match(fugue_sqlParser.T__2)
                pass
            elif token in [8]:
                localctx = fugue_sqlParser.FugueSchemaStructTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 1055
                self.match(fugue_sqlParser.T__7)
                self.state = 1056
                self.fugueSchema()
                self.state = 1057
                self.match(fugue_sqlParser.T__8)
                pass
            elif token in [315]:
                localctx = fugue_sqlParser.FugueSchemaMapTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 1059
                self.match(fugue_sqlParser.LT)
                self.state = 1060
                self.fugueSchemaType()
                self.state = 1061
                self.match(fugue_sqlParser.T__1)
                self.state = 1062
                self.fugueSchemaType()
                self.state = 1063
                self.match(fugue_sqlParser.GT)
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

    class FugueRenamePairContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def fugueSchemaKey(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaKeyContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaKeyContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueRenamePair

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueRenamePair'):
                return visitor.visitFugueRenamePair(self)
            else:
                return visitor.visitChildren(self)

    def fugueRenamePair(self):
        localctx = fugue_sqlParser.FugueRenamePairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 126, self.RULE_fugueRenamePair)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1067
            localctx.key = self.fugueSchemaKey()
            self.state = 1068
            self.match(fugue_sqlParser.T__3)
            self.state = 1069
            localctx.value = self.fugueSchemaKey()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueJsonValue(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJson

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJson'):
                return visitor.visitFugueJson(self)
            else:
                return visitor.visitChildren(self)

    def fugueJson(self):
        localctx = fugue_sqlParser.FugueJsonContext(self, self._ctx, self.state)
        self.enterRule(localctx, 128, self.RULE_fugueJson)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1071
            self.fugueJsonValue()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonObjContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueJsonPairs(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairsContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonObj

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonObj'):
                return visitor.visitFugueJsonObj(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonObj(self):
        localctx = fugue_sqlParser.FugueJsonObjContext(self, self._ctx, self.state)
        self.enterRule(localctx, 130, self.RULE_fugueJsonObj)
        self._la = 0
        try:
            self.state = 1091
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 124, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1073
                self.match(fugue_sqlParser.T__7)
                self.state = 1074
                self.fugueJsonPairs()
                self.state = 1076
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 1075
                    self.match(fugue_sqlParser.T__1)
                self.state = 1078
                self.match(fugue_sqlParser.T__8)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1080
                self.match(fugue_sqlParser.T__7)
                self.state = 1081
                self.match(fugue_sqlParser.T__8)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 1082
                self.match(fugue_sqlParser.T__4)
                self.state = 1083
                self.fugueJsonPairs()
                self.state = 1085
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 1084
                    self.match(fugue_sqlParser.T__1)
                self.state = 1087
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 1089
                self.match(fugue_sqlParser.T__4)
                self.state = 1090
                self.match(fugue_sqlParser.T__5)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonPairsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueJsonPair(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonPairContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonPairs

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonPairs'):
                return visitor.visitFugueJsonPairs(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonPairs(self):
        localctx = fugue_sqlParser.FugueJsonPairsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 132, self.RULE_fugueJsonPairs)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1093
            self.fugueJsonPair()
            self.state = 1098
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 125, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 1094
                    self.match(fugue_sqlParser.T__1)
                    self.state = 1095
                    self.fugueJsonPair()
                self.state = 1100
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 125, self._ctx)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonPairContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.key = None
            self.value = None

        def fugueJsonKey(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonKeyContext, 0)

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def fugueJsonValue(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonPair

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonPair'):
                return visitor.visitFugueJsonPair(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonPair(self):
        localctx = fugue_sqlParser.FugueJsonPairContext(self, self._ctx, self.state)
        self.enterRule(localctx, 134, self.RULE_fugueJsonPair)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1101
            localctx.key = self.fugueJsonKey()
            self.state = 1102
            _la = self._input.LA(1)
            if not (_la == 4 or _la == 310):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 1103
            localctx.value = self.fugueJsonValue()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonKeyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)

        def fugueJsonString(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonStringContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonKey

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonKey'):
                return visitor.visitFugueJsonKey(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonKey(self):
        localctx = fugue_sqlParser.FugueJsonKeyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 136, self.RULE_fugueJsonKey)
        try:
            self.state = 1107
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1105
                self.fugueIdentifier()
                pass
            elif token in [330]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1106
                self.fugueJsonString()
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

    class FugueJsonArrayContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueJsonValue(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonValueContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonArray

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonArray'):
                return visitor.visitFugueJsonArray(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonArray(self):
        localctx = fugue_sqlParser.FugueJsonArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 138, self.RULE_fugueJsonArray)
        self._la = 0
        try:
            self.state = 1125
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 129, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1109
                self.match(fugue_sqlParser.T__0)
                self.state = 1110
                self.fugueJsonValue()
                self.state = 1115
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 127, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 1111
                        self.match(fugue_sqlParser.T__1)
                        self.state = 1112
                        self.fugueJsonValue()
                    self.state = 1117
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 127, self._ctx)
                self.state = 1119
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 1118
                    self.match(fugue_sqlParser.T__1)
                self.state = 1121
                self.match(fugue_sqlParser.T__2)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1123
                self.match(fugue_sqlParser.T__0)
                self.state = 1124
                self.match(fugue_sqlParser.T__2)
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fugueJsonString(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonStringContext, 0)

        def fugueJsonNumber(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonNumberContext, 0)

        def fugueJsonObj(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonObjContext, 0)

        def fugueJsonArray(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonArrayContext, 0)

        def fugueJsonBool(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonBoolContext, 0)

        def fugueJsonNull(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueJsonNullContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonValue'):
                return visitor.visitFugueJsonValue(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonValue(self):
        localctx = fugue_sqlParser.FugueJsonValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 140, self.RULE_fugueJsonValue)
        try:
            self.state = 1133
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [330]:
                self.enterOuterAlt(localctx, 1)
                self.state = 1127
                self.fugueJsonString()
                pass
            elif token in [320, 331, 332, 333, 334, 335, 336, 337, 338]:
                self.enterOuterAlt(localctx, 2)
                self.state = 1128
                self.fugueJsonNumber()
                pass
            elif token in [5, 8]:
                self.enterOuterAlt(localctx, 3)
                self.state = 1129
                self.fugueJsonObj()
                pass
            elif token in [1]:
                self.enterOuterAlt(localctx, 4)
                self.state = 1130
                self.fugueJsonArray()
                pass
            elif token in [10, 11, 135, 287]:
                self.enterOuterAlt(localctx, 5)
                self.state = 1131
                self.fugueJsonBool()
                pass
            elif token in [12, 200]:
                self.enterOuterAlt(localctx, 6)
                self.state = 1132
                self.fugueJsonNull()
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

    class FugueJsonNumberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def number(self):
            return self.getTypedRuleContext(fugue_sqlParser.NumberContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonNumber

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonNumber'):
                return visitor.visitFugueJsonNumber(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonNumber(self):
        localctx = fugue_sqlParser.FugueJsonNumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 142, self.RULE_fugueJsonNumber)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1135
            self.number()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonStringContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonString

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonString'):
                return visitor.visitFugueJsonString(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonString(self):
        localctx = fugue_sqlParser.FugueJsonStringContext(self, self._ctx, self.state)
        self.enterRule(localctx, 144, self.RULE_fugueJsonString)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1137
            self.match(fugue_sqlParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FugueJsonBoolContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRUE(self):
            return self.getToken(fugue_sqlParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(fugue_sqlParser.FALSE, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonBool

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonBool'):
                return visitor.visitFugueJsonBool(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonBool(self):
        localctx = fugue_sqlParser.FugueJsonBoolContext(self, self._ctx, self.state)
        self.enterRule(localctx, 146, self.RULE_fugueJsonBool)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1139
            _la = self._input.LA(1)
            if not (_la == 10 or _la == 11 or _la == 135 or (_la == 287)):
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

    class FugueJsonNullContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueJsonNull

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueJsonNull'):
                return visitor.visitFugueJsonNull(self)
            else:
                return visitor.visitChildren(self)

    def fugueJsonNull(self):
        localctx = fugue_sqlParser.FugueJsonNullContext(self, self._ctx, self.state)
        self.enterRule(localctx, 148, self.RULE_fugueJsonNull)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1141
            _la = self._input.LA(1)
            if not (_la == 12 or _la == 200):
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

    class FugueIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fugueIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueIdentifier'):
                return visitor.visitFugueIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def fugueIdentifier(self):
        localctx = fugue_sqlParser.FugueIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 150, self.RULE_fugueIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1143
            self.identifier()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SingleStatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statement(self):
            return self.getTypedRuleContext(fugue_sqlParser.StatementContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleStatement'):
                return visitor.visitSingleStatement(self)
            else:
                return visitor.visitChildren(self)

    def singleStatement(self):
        localctx = fugue_sqlParser.SingleStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 152, self.RULE_singleStatement)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1145
            self.statement()
            self.state = 1149
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 13:
                self.state = 1146
                self.match(fugue_sqlParser.T__12)
                self.state = 1151
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 1152
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleExpression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleExpression'):
                return visitor.visitSingleExpression(self)
            else:
                return visitor.visitChildren(self)

    def singleExpression(self):
        localctx = fugue_sqlParser.SingleExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 154, self.RULE_singleExpression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1154
            self.namedExpression()
            self.state = 1155
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleTableIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleTableIdentifier'):
                return visitor.visitSingleTableIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleTableIdentifier(self):
        localctx = fugue_sqlParser.SingleTableIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 156, self.RULE_singleTableIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1157
            self.tableIdentifier()
            self.state = 1158
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleMultipartIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleMultipartIdentifier'):
                return visitor.visitSingleMultipartIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleMultipartIdentifier(self):
        localctx = fugue_sqlParser.SingleMultipartIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 158, self.RULE_singleMultipartIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1160
            self.multipartIdentifier()
            self.state = 1161
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.FunctionIdentifierContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleFunctionIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleFunctionIdentifier'):
                return visitor.visitSingleFunctionIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def singleFunctionIdentifier(self):
        localctx = fugue_sqlParser.SingleFunctionIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 160, self.RULE_singleFunctionIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1163
            self.functionIdentifier()
            self.state = 1164
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleDataType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleDataType'):
                return visitor.visitSingleDataType(self)
            else:
                return visitor.visitChildren(self)

    def singleDataType(self):
        localctx = fugue_sqlParser.SingleDataTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 162, self.RULE_singleDataType)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1166
            self.dataType()
            self.state = 1167
            self.match(fugue_sqlParser.EOF)
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
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

        def EOF(self):
            return self.getToken(fugue_sqlParser.EOF, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_singleTableSchema

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSingleTableSchema'):
                return visitor.visitSingleTableSchema(self)
            else:
                return visitor.visitChildren(self)

    def singleTableSchema(self):
        localctx = fugue_sqlParser.SingleTableSchemaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 164, self.RULE_singleTableSchema)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 1169
            self.colTypeList()
            self.state = 1170
            self.match(fugue_sqlParser.EOF)
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
            return fugue_sqlParser.RULE_statement

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ExplainContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXPLAIN(self):
            return self.getToken(fugue_sqlParser.EXPLAIN, 0)

        def statement(self):
            return self.getTypedRuleContext(fugue_sqlParser.StatementContext, 0)

        def LOGICAL(self):
            return self.getToken(fugue_sqlParser.LOGICAL, 0)

        def FORMATTED(self):
            return self.getToken(fugue_sqlParser.FORMATTED, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def CODEGEN(self):
            return self.getToken(fugue_sqlParser.CODEGEN, 0)

        def COST(self):
            return self.getToken(fugue_sqlParser.COST, 0)

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
            return self.getToken(fugue_sqlParser.RESET, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

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
            return self.getToken(fugue_sqlParser.USE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def NAMESPACE(self):
            return self.getToken(fugue_sqlParser.NAMESPACE, 0)

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
            return self.getToken(fugue_sqlParser.DROP, 0)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def RESTRICT(self):
            return self.getToken(fugue_sqlParser.RESTRICT, 0)

        def CASCADE(self):
            return self.getToken(fugue_sqlParser.CASCADE, 0)

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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def tableIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def GLOBAL(self):
            return self.getToken(fugue_sqlParser.GLOBAL, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

        def OPTIONS(self):
            return self.getToken(fugue_sqlParser.OPTIONS, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTempViewUsing'):
                return visitor.visitCreateTempViewUsing(self)
            else:
                return visitor.visitChildren(self)

    class RenameTableContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.ifrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

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
            return self.getToken(fugue_sqlParser.SET, 0)

        def ROLE(self):
            return self.getToken(fugue_sqlParser.ROLE, 0)

        def unsupportedHiveNativeCommands(self):
            return self.getTypedRuleContext(fugue_sqlParser.UnsupportedHiveNativeCommandsContext, 0)

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
            return self.getToken(fugue_sqlParser.CLEAR, 0)

        def CACHE(self):
            return self.getToken(fugue_sqlParser.CACHE, 0)

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
            return self.getToken(fugue_sqlParser.DROP, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def TABLES(self):
            return self.getToken(fugue_sqlParser.TABLES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def RECOVER(self):
            return self.getToken(fugue_sqlParser.RECOVER, 0)

        def PARTITIONS(self):
            return self.getToken(fugue_sqlParser.PARTITIONS, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def CURRENT(self):
            return self.getToken(fugue_sqlParser.CURRENT, 0)

        def NAMESPACE(self):
            return self.getToken(fugue_sqlParser.NAMESPACE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitShowCurrentNamespace'):
                return visitor.visitShowCurrentNamespace(self)
            else:
                return visitor.visitChildren(self)

    class RenameTablePartitionContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.ifrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def partitionSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, i)

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
            return self.getToken(fugue_sqlParser.MSCK, 0)

        def REPAIR(self):
            return self.getToken(fugue_sqlParser.REPAIR, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.REFRESH, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def CREATE(self):
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def SERDE(self):
            return self.getToken(fugue_sqlParser.SERDE, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def DATABASES(self):
            return self.getToken(fugue_sqlParser.DATABASES, 0)

        def NAMESPACES(self):
            return self.getToken(fugue_sqlParser.NAMESPACES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def FROM(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.FROM)
            else:
                return self.getToken(fugue_sqlParser.FROM, i)

        def IN(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.IN)
            else:
                return self.getToken(fugue_sqlParser.IN, i)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

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
            return self.getTypedRuleContext(fugue_sqlParser.ReplaceTableHeaderContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

        def createTableClauses(self):
            return self.getTypedRuleContext(fugue_sqlParser.CreateTableClausesContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def ADD(self):
            return self.getToken(fugue_sqlParser.ADD, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def partitionSpecLocation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecLocationContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecLocationContext, i)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.REFRESH, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def DBPROPERTIES(self):
            return self.getToken(fugue_sqlParser.DBPROPERTIES, 0)

        def PROPERTIES(self):
            return self.getToken(fugue_sqlParser.PROPERTIES, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def ADD(self):
            return self.getToken(fugue_sqlParser.ADD, 0)

        def LIST(self):
            return self.getToken(fugue_sqlParser.LIST, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

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
            return self.getToken(fugue_sqlParser.ANALYZE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def COMPUTE(self):
            return self.getToken(fugue_sqlParser.COMPUTE, 0)

        def STATISTICS(self):
            return self.getToken(fugue_sqlParser.STATISTICS, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def FOR(self):
            return self.getToken(fugue_sqlParser.FOR, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def identifierSeq(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.CreateTableHeaderContext, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

        def bucketSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.BucketSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.BucketSpecContext, i)

        def skewSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.SkewSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.SkewSpecContext, i)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

        def createFileFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CreateFileFormatContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def colTypeList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ColTypeListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.PARTITIONED)
            else:
                return self.getToken(fugue_sqlParser.PARTITIONED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

        def identifierList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def FUNCTION(self):
            return self.getToken(fugue_sqlParser.FUNCTION, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def resource(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ResourceContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ResourceContext, i)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def qualifiedColTypeWithPositionList(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionListContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.COMMENT, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IS(self):
            return self.getToken(fugue_sqlParser.IS, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.CreateTableHeaderContext, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

        def createTableClauses(self):
            return self.getTypedRuleContext(fugue_sqlParser.CreateTableClausesContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.DmlStatementNoWithContext, 0)

        def ctes(self):
            return self.getTypedRuleContext(fugue_sqlParser.CtesContext, 0)

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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def tableIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TableIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, i)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def tableProvider(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TableProviderContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, i)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

        def createFileFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CreateFileFormatContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

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
            return self.getToken(fugue_sqlParser.UNCACHE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

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
            return self.getToken(fugue_sqlParser.DROP, 0)

        def FUNCTION(self):
            return self.getToken(fugue_sqlParser.FUNCTION, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def describeColName(self):
            return self.getTypedRuleContext(fugue_sqlParser.DescribeColNameContext, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def FORMATTED(self):
            return self.getToken(fugue_sqlParser.FORMATTED, 0)

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
            return self.getToken(fugue_sqlParser.LOAD, 0)

        def DATA(self):
            return self.getToken(fugue_sqlParser.DATA, 0)

        def INPATH(self):
            return self.getToken(fugue_sqlParser.INPATH, 0)

        def INTO(self):
            return self.getToken(fugue_sqlParser.INTO, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def PARTITIONS(self):
            return self.getToken(fugue_sqlParser.PARTITIONS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.FUNCTION, 0)

        def describeFuncName(self):
            return self.getTypedRuleContext(fugue_sqlParser.DescribeFuncNameContext, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeFunction'):
                return visitor.visitDescribeFunction(self)
            else:
                return visitor.visitChildren(self)

    class RenameTableColumnContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.table = None
            self.ifrom = None
            self.to = None
            self.copyFrom(ctx)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def CHANGE(self):
            return self.getToken(fugue_sqlParser.CHANGE, 0)

        def colType(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeContext, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def colPosition(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def QUERY(self):
            return self.getToken(fugue_sqlParser.QUERY, 0)

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
            return self.getToken(fugue_sqlParser.TRUNCATE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def SERDE(self):
            return self.getToken(fugue_sqlParser.SERDE, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def WITH(self):
            return self.getToken(fugue_sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def identifierCommentList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierCommentListContext, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.PARTITIONED)
            else:
                return self.getToken(fugue_sqlParser.PARTITIONED, i)

        def ON(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.ON)
            else:
                return self.getToken(fugue_sqlParser.ON, i)

        def identifierList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

        def GLOBAL(self):
            return self.getToken(fugue_sqlParser.GLOBAL, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def partitionSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PartitionSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, i)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def PURGE(self):
            return self.getToken(fugue_sqlParser.PURGE, 0)

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
            return self.getToken(fugue_sqlParser.SET, 0)

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
            return self.getToken(fugue_sqlParser.DROP, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def PURGE(self):
            return self.getToken(fugue_sqlParser.PURGE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDropTable'):
                return visitor.visitDropTable(self)
            else:
                return visitor.visitChildren(self)

    class DescribeNamespaceContext(StatementContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

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
                return self.getTokens(fugue_sqlParser.ALTER)
            else:
                return self.getToken(fugue_sqlParser.ALTER, i)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

        def CHANGE(self):
            return self.getToken(fugue_sqlParser.CHANGE, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def alterColumnAction(self):
            return self.getTypedRuleContext(fugue_sqlParser.AlterColumnActionContext, 0)

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
            return self.getToken(fugue_sqlParser.COMMENT, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IS(self):
            return self.getToken(fugue_sqlParser.IS, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def theNamespace(self):
            return self.getTypedRuleContext(fugue_sqlParser.TheNamespaceContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

        def WITH(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.WITH)
            else:
                return self.getToken(fugue_sqlParser.WITH, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

        def DBPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.DBPROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.DBPROPERTIES, i)

        def PROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.PROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.PROPERTIES, i)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def TBLPROPERTIES(self):
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def tablePropertyKey(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyKeyContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def UNSET(self):
            return self.getToken(fugue_sqlParser.UNSET, 0)

        def TBLPROPERTIES(self):
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def multipartIdentifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierListContext, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def VIEWS(self):
            return self.getToken(fugue_sqlParser.VIEWS, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

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
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def FUNCTIONS(self):
            return self.getToken(fugue_sqlParser.FUNCTIONS, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

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
            return self.getToken(fugue_sqlParser.CACHE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def OPTIONS(self):
            return self.getToken(fugue_sqlParser.OPTIONS, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def ADD(self):
            return self.getToken(fugue_sqlParser.ADD, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def qualifiedColTypeWithPositionList(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionListContext, 0)

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
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def TBLPROPERTIES(self):
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetTableProperties'):
                return visitor.visitSetTableProperties(self)
            else:
                return visitor.visitChildren(self)

    def statement(self):
        localctx = fugue_sqlParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 166, self.RULE_statement)
        self._la = 0
        try:
            self.state = 1877
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 237, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.StatementDefaultContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 1172
                self.query()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.DmlStatementContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 1174
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 308:
                    self.state = 1173
                    self.ctes()
                self.state = 1176
                self.dmlStatementNoWith()
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.UseContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 1177
                self.match(fugue_sqlParser.USE)
                self.state = 1179
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 133, self._ctx)
                if la_ == 1:
                    self.state = 1178
                    self.match(fugue_sqlParser.NAMESPACE)
                self.state = 1181
                self.multipartIdentifier()
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.CreateNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 1182
                self.match(fugue_sqlParser.CREATE)
                self.state = 1183
                self.theNamespace()
                self.state = 1187
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 134, self._ctx)
                if la_ == 1:
                    self.state = 1184
                    self.match(fugue_sqlParser.IF)
                    self.state = 1185
                    self.match(fugue_sqlParser.NOT)
                    self.state = 1186
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1189
                self.multipartIdentifier()
                self.state = 1197
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 91 or _la == 184 or _la == 308:
                    self.state = 1195
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [91]:
                        self.state = 1190
                        self.commentSpec()
                        pass
                    elif token in [184]:
                        self.state = 1191
                        self.locationSpec()
                        pass
                    elif token in [308]:
                        self.state = 1192
                        self.match(fugue_sqlParser.WITH)
                        self.state = 1193
                        _la = self._input.LA(1)
                        if not (_la == 111 or _la == 226):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 1194
                        self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 1199
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif la_ == 5:
                localctx = fugue_sqlParser.SetNamespacePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 1200
                self.match(fugue_sqlParser.ALTER)
                self.state = 1201
                self.theNamespace()
                self.state = 1202
                self.multipartIdentifier()
                self.state = 1203
                self.match(fugue_sqlParser.SET)
                self.state = 1204
                _la = self._input.LA(1)
                if not (_la == 111 or _la == 226):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1205
                self.tablePropertyList()
                pass
            elif la_ == 6:
                localctx = fugue_sqlParser.SetNamespaceLocationContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 1207
                self.match(fugue_sqlParser.ALTER)
                self.state = 1208
                self.theNamespace()
                self.state = 1209
                self.multipartIdentifier()
                self.state = 1210
                self.match(fugue_sqlParser.SET)
                self.state = 1211
                self.locationSpec()
                pass
            elif la_ == 7:
                localctx = fugue_sqlParser.DropNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 1213
                self.match(fugue_sqlParser.DROP)
                self.state = 1214
                self.theNamespace()
                self.state = 1217
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 137, self._ctx)
                if la_ == 1:
                    self.state = 1215
                    self.match(fugue_sqlParser.IF)
                    self.state = 1216
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1219
                self.multipartIdentifier()
                self.state = 1221
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 78 or _la == 240:
                    self.state = 1220
                    _la = self._input.LA(1)
                    if not (_la == 78 or _la == 240):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                pass
            elif la_ == 8:
                localctx = fugue_sqlParser.ShowNamespacesContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 1223
                self.match(fugue_sqlParser.SHOW)
                self.state = 1224
                _la = self._input.LA(1)
                if not (_la == 109 or _la == 196):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1227
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 146 or _la == 159:
                    self.state = 1225
                    _la = self._input.LA(1)
                    if not (_la == 146 or _la == 159):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 1226
                    self.multipartIdentifier()
                self.state = 1233
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 178 or _la == 330:
                    self.state = 1230
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 178:
                        self.state = 1229
                        self.match(fugue_sqlParser.LIKE)
                    self.state = 1232
                    localctx.pattern = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 9:
                localctx = fugue_sqlParser.CreateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 1235
                self.createTableHeader()
                self.state = 1240
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1236
                    self.match(fugue_sqlParser.T__4)
                    self.state = 1237
                    self.colTypeList()
                    self.state = 1238
                    self.match(fugue_sqlParser.T__5)
                self.state = 1242
                self.tableProvider()
                self.state = 1243
                self.createTableClauses()
                self.state = 1248
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 17 & ~63 == 0 and 1 << _la - 17 & 2269395221545987 != 0 or (_la - 99 & ~63 == 0 and 1 << _la - 99 & 140737496743937 != 0) or (_la - 182 & ~63 == 0 and 1 << _la - 182 & 20266198323167361 != 0) or (_la - 252 & ~63 == 0 and 1 << _la - 252 & 73183502536802305 != 0):
                    self.state = 1245
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 68:
                        self.state = 1244
                        self.match(fugue_sqlParser.AS)
                    self.state = 1247
                    self.query()
                pass
            elif la_ == 10:
                localctx = fugue_sqlParser.CreateHiveTableContext(self, localctx)
                self.enterOuterAlt(localctx, 10)
                self.state = 1250
                self.createTableHeader()
                self.state = 1255
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1251
                    self.match(fugue_sqlParser.T__4)
                    self.state = 1252
                    localctx.columns = self.colTypeList()
                    self.state = 1253
                    self.match(fugue_sqlParser.T__5)
                self.state = 1278
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 85 or _la == 91 or _la == 184 or (_la == 217) or (_la - 248 & ~63 == 0 and 1 << _la - 248 & 269500417 != 0):
                    self.state = 1276
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [91]:
                        self.state = 1257
                        self.commentSpec()
                        pass
                    elif token in [217]:
                        self.state = 1267
                        self._errHandler.sync(self)
                        la_ = self._interp.adaptivePredict(self._input, 146, self._ctx)
                        if la_ == 1:
                            self.state = 1258
                            self.match(fugue_sqlParser.PARTITIONED)
                            self.state = 1259
                            self.match(fugue_sqlParser.BY)
                            self.state = 1260
                            self.match(fugue_sqlParser.T__4)
                            self.state = 1261
                            localctx.partitionColumns = self.colTypeList()
                            self.state = 1262
                            self.match(fugue_sqlParser.T__5)
                            pass
                        elif la_ == 2:
                            self.state = 1264
                            self.match(fugue_sqlParser.PARTITIONED)
                            self.state = 1265
                            self.match(fugue_sqlParser.BY)
                            self.state = 1266
                            localctx.partitionColumnNames = self.identifierList()
                            pass
                        pass
                    elif token in [85]:
                        self.state = 1269
                        self.bucketSpec()
                        pass
                    elif token in [262]:
                        self.state = 1270
                        self.skewSpec()
                        pass
                    elif token in [248]:
                        self.state = 1271
                        self.rowFormat()
                        pass
                    elif token in [268]:
                        self.state = 1272
                        self.createFileFormat()
                        pass
                    elif token in [184]:
                        self.state = 1273
                        self.locationSpec()
                        pass
                    elif token in [276]:
                        self.state = 1274
                        self.match(fugue_sqlParser.TBLPROPERTIES)
                        self.state = 1275
                        localctx.tableProps = self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 1280
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1285
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 17 & ~63 == 0 and 1 << _la - 17 & 2269395221545987 != 0 or (_la - 99 & ~63 == 0 and 1 << _la - 99 & 140737496743937 != 0) or (_la - 182 & ~63 == 0 and 1 << _la - 182 & 20266198323167361 != 0) or (_la - 252 & ~63 == 0 and 1 << _la - 252 & 73183502536802305 != 0):
                    self.state = 1282
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 68:
                        self.state = 1281
                        self.match(fugue_sqlParser.AS)
                    self.state = 1284
                    self.query()
                pass
            elif la_ == 11:
                localctx = fugue_sqlParser.CreateTableLikeContext(self, localctx)
                self.enterOuterAlt(localctx, 11)
                self.state = 1287
                self.match(fugue_sqlParser.CREATE)
                self.state = 1288
                self.match(fugue_sqlParser.TABLE)
                self.state = 1292
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 151, self._ctx)
                if la_ == 1:
                    self.state = 1289
                    self.match(fugue_sqlParser.IF)
                    self.state = 1290
                    self.match(fugue_sqlParser.NOT)
                    self.state = 1291
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1294
                localctx.target = self.tableIdentifier()
                self.state = 1295
                self.match(fugue_sqlParser.LIKE)
                self.state = 1296
                localctx.source = self.tableIdentifier()
                self.state = 1305
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 184 or (_la - 248 & ~63 == 0 and 1 << _la - 248 & 9007199524225025 != 0):
                    self.state = 1303
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [301]:
                        self.state = 1297
                        self.tableProvider()
                        pass
                    elif token in [248]:
                        self.state = 1298
                        self.rowFormat()
                        pass
                    elif token in [268]:
                        self.state = 1299
                        self.createFileFormat()
                        pass
                    elif token in [184]:
                        self.state = 1300
                        self.locationSpec()
                        pass
                    elif token in [276]:
                        self.state = 1301
                        self.match(fugue_sqlParser.TBLPROPERTIES)
                        self.state = 1302
                        localctx.tableProps = self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 1307
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif la_ == 12:
                localctx = fugue_sqlParser.ReplaceTableContext(self, localctx)
                self.enterOuterAlt(localctx, 12)
                self.state = 1308
                self.replaceTableHeader()
                self.state = 1313
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1309
                    self.match(fugue_sqlParser.T__4)
                    self.state = 1310
                    self.colTypeList()
                    self.state = 1311
                    self.match(fugue_sqlParser.T__5)
                self.state = 1315
                self.tableProvider()
                self.state = 1316
                self.createTableClauses()
                self.state = 1321
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 17 & ~63 == 0 and 1 << _la - 17 & 2269395221545987 != 0 or (_la - 99 & ~63 == 0 and 1 << _la - 99 & 140737496743937 != 0) or (_la - 182 & ~63 == 0 and 1 << _la - 182 & 20266198323167361 != 0) or (_la - 252 & ~63 == 0 and 1 << _la - 252 & 73183502536802305 != 0):
                    self.state = 1318
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 68:
                        self.state = 1317
                        self.match(fugue_sqlParser.AS)
                    self.state = 1320
                    self.query()
                pass
            elif la_ == 13:
                localctx = fugue_sqlParser.AnalyzeContext(self, localctx)
                self.enterOuterAlt(localctx, 13)
                self.state = 1323
                self.match(fugue_sqlParser.ANALYZE)
                self.state = 1324
                self.match(fugue_sqlParser.TABLE)
                self.state = 1325
                self.multipartIdentifier()
                self.state = 1327
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1326
                    self.partitionSpec()
                self.state = 1329
                self.match(fugue_sqlParser.COMPUTE)
                self.state = 1330
                self.match(fugue_sqlParser.STATISTICS)
                self.state = 1338
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 158, self._ctx)
                if la_ == 1:
                    self.state = 1331
                    self.identifier()
                elif la_ == 2:
                    self.state = 1332
                    self.match(fugue_sqlParser.FOR)
                    self.state = 1333
                    self.match(fugue_sqlParser.COLUMNS)
                    self.state = 1334
                    self.identifierSeq()
                elif la_ == 3:
                    self.state = 1335
                    self.match(fugue_sqlParser.FOR)
                    self.state = 1336
                    self.match(fugue_sqlParser.ALL)
                    self.state = 1337
                    self.match(fugue_sqlParser.COLUMNS)
                pass
            elif la_ == 14:
                localctx = fugue_sqlParser.AddTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 14)
                self.state = 1340
                self.match(fugue_sqlParser.ALTER)
                self.state = 1341
                self.match(fugue_sqlParser.TABLE)
                self.state = 1342
                self.multipartIdentifier()
                self.state = 1343
                self.match(fugue_sqlParser.ADD)
                self.state = 1344
                _la = self._input.LA(1)
                if not (_la == 89 or _la == 90):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1345
                localctx.columns = self.qualifiedColTypeWithPositionList()
                pass
            elif la_ == 15:
                localctx = fugue_sqlParser.AddTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 15)
                self.state = 1347
                self.match(fugue_sqlParser.ALTER)
                self.state = 1348
                self.match(fugue_sqlParser.TABLE)
                self.state = 1349
                self.multipartIdentifier()
                self.state = 1350
                self.match(fugue_sqlParser.ADD)
                self.state = 1351
                _la = self._input.LA(1)
                if not (_la == 89 or _la == 90):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1352
                self.match(fugue_sqlParser.T__4)
                self.state = 1353
                localctx.columns = self.qualifiedColTypeWithPositionList()
                self.state = 1354
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 16:
                localctx = fugue_sqlParser.RenameTableColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 16)
                self.state = 1356
                self.match(fugue_sqlParser.ALTER)
                self.state = 1357
                self.match(fugue_sqlParser.TABLE)
                self.state = 1358
                localctx.table = self.multipartIdentifier()
                self.state = 1359
                self.match(fugue_sqlParser.RENAME)
                self.state = 1360
                self.match(fugue_sqlParser.COLUMN)
                self.state = 1361
                localctx.ifrom = self.multipartIdentifier()
                self.state = 1362
                self.match(fugue_sqlParser.TO)
                self.state = 1363
                localctx.to = self.errorCapturingIdentifier()
                pass
            elif la_ == 17:
                localctx = fugue_sqlParser.DropTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 17)
                self.state = 1365
                self.match(fugue_sqlParser.ALTER)
                self.state = 1366
                self.match(fugue_sqlParser.TABLE)
                self.state = 1367
                self.multipartIdentifier()
                self.state = 1368
                self.match(fugue_sqlParser.DROP)
                self.state = 1369
                _la = self._input.LA(1)
                if not (_la == 89 or _la == 90):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1370
                self.match(fugue_sqlParser.T__4)
                self.state = 1371
                localctx.columns = self.multipartIdentifierList()
                self.state = 1372
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 18:
                localctx = fugue_sqlParser.DropTableColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 18)
                self.state = 1374
                self.match(fugue_sqlParser.ALTER)
                self.state = 1375
                self.match(fugue_sqlParser.TABLE)
                self.state = 1376
                self.multipartIdentifier()
                self.state = 1377
                self.match(fugue_sqlParser.DROP)
                self.state = 1378
                _la = self._input.LA(1)
                if not (_la == 89 or _la == 90):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1379
                localctx.columns = self.multipartIdentifierList()
                pass
            elif la_ == 19:
                localctx = fugue_sqlParser.RenameTableContext(self, localctx)
                self.enterOuterAlt(localctx, 19)
                self.state = 1381
                self.match(fugue_sqlParser.ALTER)
                self.state = 1382
                _la = self._input.LA(1)
                if not (_la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1383
                localctx.ifrom = self.multipartIdentifier()
                self.state = 1384
                self.match(fugue_sqlParser.RENAME)
                self.state = 1385
                self.match(fugue_sqlParser.TO)
                self.state = 1386
                localctx.to = self.multipartIdentifier()
                pass
            elif la_ == 20:
                localctx = fugue_sqlParser.SetTablePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 20)
                self.state = 1388
                self.match(fugue_sqlParser.ALTER)
                self.state = 1389
                _la = self._input.LA(1)
                if not (_la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1390
                self.multipartIdentifier()
                self.state = 1391
                self.match(fugue_sqlParser.SET)
                self.state = 1392
                self.match(fugue_sqlParser.TBLPROPERTIES)
                self.state = 1393
                self.tablePropertyList()
                pass
            elif la_ == 21:
                localctx = fugue_sqlParser.UnsetTablePropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 21)
                self.state = 1395
                self.match(fugue_sqlParser.ALTER)
                self.state = 1396
                _la = self._input.LA(1)
                if not (_la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1397
                self.multipartIdentifier()
                self.state = 1398
                self.match(fugue_sqlParser.UNSET)
                self.state = 1399
                self.match(fugue_sqlParser.TBLPROPERTIES)
                self.state = 1402
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 156:
                    self.state = 1400
                    self.match(fugue_sqlParser.IF)
                    self.state = 1401
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1404
                self.tablePropertyList()
                pass
            elif la_ == 22:
                localctx = fugue_sqlParser.AlterTableAlterColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 22)
                self.state = 1406
                self.match(fugue_sqlParser.ALTER)
                self.state = 1407
                self.match(fugue_sqlParser.TABLE)
                self.state = 1408
                localctx.table = self.multipartIdentifier()
                self.state = 1409
                _la = self._input.LA(1)
                if not (_la == 61 or _la == 81):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1411
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 160, self._ctx)
                if la_ == 1:
                    self.state = 1410
                    self.match(fugue_sqlParser.COLUMN)
                self.state = 1413
                localctx.column = self.multipartIdentifier()
                self.state = 1415
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 59 or (_la - 91 & ~63 == 0 and 1 << _la - 91 & 562952100904961 != 0) or _la == 258 or (_la == 289):
                    self.state = 1414
                    self.alterColumnAction()
                pass
            elif la_ == 23:
                localctx = fugue_sqlParser.HiveChangeColumnContext(self, localctx)
                self.enterOuterAlt(localctx, 23)
                self.state = 1417
                self.match(fugue_sqlParser.ALTER)
                self.state = 1418
                self.match(fugue_sqlParser.TABLE)
                self.state = 1419
                localctx.table = self.multipartIdentifier()
                self.state = 1421
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1420
                    self.partitionSpec()
                self.state = 1423
                self.match(fugue_sqlParser.CHANGE)
                self.state = 1425
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 163, self._ctx)
                if la_ == 1:
                    self.state = 1424
                    self.match(fugue_sqlParser.COLUMN)
                self.state = 1427
                localctx.colName = self.multipartIdentifier()
                self.state = 1428
                self.colType()
                self.state = 1430
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 59 or _la == 140:
                    self.state = 1429
                    self.colPosition()
                pass
            elif la_ == 24:
                localctx = fugue_sqlParser.HiveReplaceColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 24)
                self.state = 1432
                self.match(fugue_sqlParser.ALTER)
                self.state = 1433
                self.match(fugue_sqlParser.TABLE)
                self.state = 1434
                localctx.table = self.multipartIdentifier()
                self.state = 1436
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1435
                    self.partitionSpec()
                self.state = 1438
                self.match(fugue_sqlParser.REPLACE)
                self.state = 1439
                self.match(fugue_sqlParser.COLUMNS)
                self.state = 1440
                self.match(fugue_sqlParser.T__4)
                self.state = 1441
                localctx.columns = self.qualifiedColTypeWithPositionList()
                self.state = 1442
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 25:
                localctx = fugue_sqlParser.SetTableSerDeContext(self, localctx)
                self.enterOuterAlt(localctx, 25)
                self.state = 1444
                self.match(fugue_sqlParser.ALTER)
                self.state = 1445
                self.match(fugue_sqlParser.TABLE)
                self.state = 1446
                self.multipartIdentifier()
                self.state = 1448
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1447
                    self.partitionSpec()
                self.state = 1450
                self.match(fugue_sqlParser.SET)
                self.state = 1451
                self.match(fugue_sqlParser.SERDE)
                self.state = 1452
                self.match(fugue_sqlParser.STRING)
                self.state = 1456
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 308:
                    self.state = 1453
                    self.match(fugue_sqlParser.WITH)
                    self.state = 1454
                    self.match(fugue_sqlParser.SERDEPROPERTIES)
                    self.state = 1455
                    self.tablePropertyList()
                pass
            elif la_ == 26:
                localctx = fugue_sqlParser.SetTableSerDeContext(self, localctx)
                self.enterOuterAlt(localctx, 26)
                self.state = 1458
                self.match(fugue_sqlParser.ALTER)
                self.state = 1459
                self.match(fugue_sqlParser.TABLE)
                self.state = 1460
                self.multipartIdentifier()
                self.state = 1462
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1461
                    self.partitionSpec()
                self.state = 1464
                self.match(fugue_sqlParser.SET)
                self.state = 1465
                self.match(fugue_sqlParser.SERDEPROPERTIES)
                self.state = 1466
                self.tablePropertyList()
                pass
            elif la_ == 27:
                localctx = fugue_sqlParser.AddTablePartitionContext(self, localctx)
                self.enterOuterAlt(localctx, 27)
                self.state = 1468
                self.match(fugue_sqlParser.ALTER)
                self.state = 1469
                _la = self._input.LA(1)
                if not (_la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1470
                self.multipartIdentifier()
                self.state = 1471
                self.match(fugue_sqlParser.ADD)
                self.state = 1475
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 156:
                    self.state = 1472
                    self.match(fugue_sqlParser.IF)
                    self.state = 1473
                    self.match(fugue_sqlParser.NOT)
                    self.state = 1474
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1478
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 1477
                    self.partitionSpecLocation()
                    self.state = 1480
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 216:
                        break
                pass
            elif la_ == 28:
                localctx = fugue_sqlParser.RenameTablePartitionContext(self, localctx)
                self.enterOuterAlt(localctx, 28)
                self.state = 1482
                self.match(fugue_sqlParser.ALTER)
                self.state = 1483
                self.match(fugue_sqlParser.TABLE)
                self.state = 1484
                self.multipartIdentifier()
                self.state = 1485
                localctx.ifrom = self.partitionSpec()
                self.state = 1486
                self.match(fugue_sqlParser.RENAME)
                self.state = 1487
                self.match(fugue_sqlParser.TO)
                self.state = 1488
                localctx.to = self.partitionSpec()
                pass
            elif la_ == 29:
                localctx = fugue_sqlParser.DropTablePartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 29)
                self.state = 1490
                self.match(fugue_sqlParser.ALTER)
                self.state = 1491
                _la = self._input.LA(1)
                if not (_la == 273 or _la == 303):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1492
                self.multipartIdentifier()
                self.state = 1493
                self.match(fugue_sqlParser.DROP)
                self.state = 1496
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 156:
                    self.state = 1494
                    self.match(fugue_sqlParser.IF)
                    self.state = 1495
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1498
                self.partitionSpec()
                self.state = 1503
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 1499
                    self.match(fugue_sqlParser.T__1)
                    self.state = 1500
                    self.partitionSpec()
                    self.state = 1505
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1507
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 227:
                    self.state = 1506
                    self.match(fugue_sqlParser.PURGE)
                pass
            elif la_ == 30:
                localctx = fugue_sqlParser.SetTableLocationContext(self, localctx)
                self.enterOuterAlt(localctx, 30)
                self.state = 1509
                self.match(fugue_sqlParser.ALTER)
                self.state = 1510
                self.match(fugue_sqlParser.TABLE)
                self.state = 1511
                self.multipartIdentifier()
                self.state = 1513
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1512
                    self.partitionSpec()
                self.state = 1515
                self.match(fugue_sqlParser.SET)
                self.state = 1516
                self.locationSpec()
                pass
            elif la_ == 31:
                localctx = fugue_sqlParser.RecoverPartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 31)
                self.state = 1518
                self.match(fugue_sqlParser.ALTER)
                self.state = 1519
                self.match(fugue_sqlParser.TABLE)
                self.state = 1520
                self.multipartIdentifier()
                self.state = 1521
                self.match(fugue_sqlParser.RECOVER)
                self.state = 1522
                self.match(fugue_sqlParser.PARTITIONS)
                pass
            elif la_ == 32:
                localctx = fugue_sqlParser.DropTableContext(self, localctx)
                self.enterOuterAlt(localctx, 32)
                self.state = 1524
                self.match(fugue_sqlParser.DROP)
                self.state = 1525
                self.match(fugue_sqlParser.TABLE)
                self.state = 1528
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 175, self._ctx)
                if la_ == 1:
                    self.state = 1526
                    self.match(fugue_sqlParser.IF)
                    self.state = 1527
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1530
                self.multipartIdentifier()
                self.state = 1532
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 227:
                    self.state = 1531
                    self.match(fugue_sqlParser.PURGE)
                pass
            elif la_ == 33:
                localctx = fugue_sqlParser.DropViewContext(self, localctx)
                self.enterOuterAlt(localctx, 33)
                self.state = 1534
                self.match(fugue_sqlParser.DROP)
                self.state = 1535
                self.match(fugue_sqlParser.VIEW)
                self.state = 1538
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 177, self._ctx)
                if la_ == 1:
                    self.state = 1536
                    self.match(fugue_sqlParser.IF)
                    self.state = 1537
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1540
                self.multipartIdentifier()
                pass
            elif la_ == 34:
                localctx = fugue_sqlParser.CreateViewContext(self, localctx)
                self.enterOuterAlt(localctx, 34)
                self.state = 1541
                self.match(fugue_sqlParser.CREATE)
                self.state = 1544
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 207:
                    self.state = 1542
                    self.match(fugue_sqlParser.OR)
                    self.state = 1543
                    self.match(fugue_sqlParser.REPLACE)
                self.state = 1550
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 150 or _la == 277:
                    self.state = 1547
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 150:
                        self.state = 1546
                        self.match(fugue_sqlParser.GLOBAL)
                    self.state = 1549
                    self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1552
                self.match(fugue_sqlParser.VIEW)
                self.state = 1556
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 181, self._ctx)
                if la_ == 1:
                    self.state = 1553
                    self.match(fugue_sqlParser.IF)
                    self.state = 1554
                    self.match(fugue_sqlParser.NOT)
                    self.state = 1555
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1558
                self.multipartIdentifier()
                self.state = 1560
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1559
                    self.identifierCommentList()
                self.state = 1570
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 91 or _la == 217 or _la == 276:
                    self.state = 1568
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [91]:
                        self.state = 1562
                        self.commentSpec()
                        pass
                    elif token in [217]:
                        self.state = 1563
                        self.match(fugue_sqlParser.PARTITIONED)
                        self.state = 1564
                        self.match(fugue_sqlParser.ON)
                        self.state = 1565
                        self.identifierList()
                        pass
                    elif token in [276]:
                        self.state = 1566
                        self.match(fugue_sqlParser.TBLPROPERTIES)
                        self.state = 1567
                        self.tablePropertyList()
                        pass
                    else:
                        raise NoViableAltException(self)
                    self.state = 1572
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 1573
                self.match(fugue_sqlParser.AS)
                self.state = 1574
                self.query()
                pass
            elif la_ == 35:
                localctx = fugue_sqlParser.CreateTempViewUsingContext(self, localctx)
                self.enterOuterAlt(localctx, 35)
                self.state = 1576
                self.match(fugue_sqlParser.CREATE)
                self.state = 1579
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 207:
                    self.state = 1577
                    self.match(fugue_sqlParser.OR)
                    self.state = 1578
                    self.match(fugue_sqlParser.REPLACE)
                self.state = 1582
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 150:
                    self.state = 1581
                    self.match(fugue_sqlParser.GLOBAL)
                self.state = 1584
                self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1585
                self.match(fugue_sqlParser.VIEW)
                self.state = 1586
                self.tableIdentifier()
                self.state = 1591
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1587
                    self.match(fugue_sqlParser.T__4)
                    self.state = 1588
                    self.colTypeList()
                    self.state = 1589
                    self.match(fugue_sqlParser.T__5)
                self.state = 1593
                self.tableProvider()
                self.state = 1596
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 206:
                    self.state = 1594
                    self.match(fugue_sqlParser.OPTIONS)
                    self.state = 1595
                    self.tablePropertyList()
                pass
            elif la_ == 36:
                localctx = fugue_sqlParser.AlterViewQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 36)
                self.state = 1598
                self.match(fugue_sqlParser.ALTER)
                self.state = 1599
                self.match(fugue_sqlParser.VIEW)
                self.state = 1600
                self.multipartIdentifier()
                self.state = 1602
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 68:
                    self.state = 1601
                    self.match(fugue_sqlParser.AS)
                self.state = 1604
                self.query()
                pass
            elif la_ == 37:
                localctx = fugue_sqlParser.CreateFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 37)
                self.state = 1606
                self.match(fugue_sqlParser.CREATE)
                self.state = 1609
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 207:
                    self.state = 1607
                    self.match(fugue_sqlParser.OR)
                    self.state = 1608
                    self.match(fugue_sqlParser.REPLACE)
                self.state = 1612
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 277:
                    self.state = 1611
                    self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1614
                self.match(fugue_sqlParser.FUNCTION)
                self.state = 1618
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 192, self._ctx)
                if la_ == 1:
                    self.state = 1615
                    self.match(fugue_sqlParser.IF)
                    self.state = 1616
                    self.match(fugue_sqlParser.NOT)
                    self.state = 1617
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1620
                self.multipartIdentifier()
                self.state = 1621
                self.match(fugue_sqlParser.AS)
                self.state = 1622
                localctx.className = self.match(fugue_sqlParser.STRING)
                self.state = 1632
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 301:
                    self.state = 1623
                    self.match(fugue_sqlParser.USING)
                    self.state = 1624
                    self.resource()
                    self.state = 1629
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 1625
                        self.match(fugue_sqlParser.T__1)
                        self.state = 1626
                        self.resource()
                        self.state = 1631
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                pass
            elif la_ == 38:
                localctx = fugue_sqlParser.DropFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 38)
                self.state = 1634
                self.match(fugue_sqlParser.DROP)
                self.state = 1636
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 277:
                    self.state = 1635
                    self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1638
                self.match(fugue_sqlParser.FUNCTION)
                self.state = 1641
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 196, self._ctx)
                if la_ == 1:
                    self.state = 1639
                    self.match(fugue_sqlParser.IF)
                    self.state = 1640
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1643
                self.multipartIdentifier()
                pass
            elif la_ == 39:
                localctx = fugue_sqlParser.ExplainContext(self, localctx)
                self.enterOuterAlt(localctx, 39)
                self.state = 1644
                self.match(fugue_sqlParser.EXPLAIN)
                self.state = 1646
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 86 & ~63 == 0 and 1 << _la - 86 & 576531121047605249 != 0 or _la == 187:
                    self.state = 1645
                    _la = self._input.LA(1)
                    if not (_la - 86 & ~63 == 0 and 1 << _la - 86 & 576531121047605249 != 0 or _la == 187):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 1648
                self.statement()
                pass
            elif la_ == 40:
                localctx = fugue_sqlParser.ShowTablesContext(self, localctx)
                self.enterOuterAlt(localctx, 40)
                self.state = 1649
                self.match(fugue_sqlParser.SHOW)
                self.state = 1650
                self.match(fugue_sqlParser.TABLES)
                self.state = 1653
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 146 or _la == 159:
                    self.state = 1651
                    _la = self._input.LA(1)
                    if not (_la == 146 or _la == 159):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 1652
                    self.multipartIdentifier()
                self.state = 1659
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 178 or _la == 330:
                    self.state = 1656
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 178:
                        self.state = 1655
                        self.match(fugue_sqlParser.LIKE)
                    self.state = 1658
                    localctx.pattern = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 41:
                localctx = fugue_sqlParser.ShowTableContext(self, localctx)
                self.enterOuterAlt(localctx, 41)
                self.state = 1661
                self.match(fugue_sqlParser.SHOW)
                self.state = 1662
                self.match(fugue_sqlParser.TABLE)
                self.state = 1663
                self.match(fugue_sqlParser.EXTENDED)
                self.state = 1666
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 146 or _la == 159:
                    self.state = 1664
                    _la = self._input.LA(1)
                    if not (_la == 146 or _la == 159):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 1665
                    localctx.ns = self.multipartIdentifier()
                self.state = 1668
                self.match(fugue_sqlParser.LIKE)
                self.state = 1669
                localctx.pattern = self.match(fugue_sqlParser.STRING)
                self.state = 1671
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1670
                    self.partitionSpec()
                pass
            elif la_ == 42:
                localctx = fugue_sqlParser.ShowTblPropertiesContext(self, localctx)
                self.enterOuterAlt(localctx, 42)
                self.state = 1673
                self.match(fugue_sqlParser.SHOW)
                self.state = 1674
                self.match(fugue_sqlParser.TBLPROPERTIES)
                self.state = 1675
                localctx.table = self.multipartIdentifier()
                self.state = 1680
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 5:
                    self.state = 1676
                    self.match(fugue_sqlParser.T__4)
                    self.state = 1677
                    localctx.key = self.tablePropertyKey()
                    self.state = 1678
                    self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 43:
                localctx = fugue_sqlParser.ShowColumnsContext(self, localctx)
                self.enterOuterAlt(localctx, 43)
                self.state = 1682
                self.match(fugue_sqlParser.SHOW)
                self.state = 1683
                self.match(fugue_sqlParser.COLUMNS)
                self.state = 1684
                _la = self._input.LA(1)
                if not (_la == 146 or _la == 159):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1685
                localctx.table = self.multipartIdentifier()
                self.state = 1688
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 146 or _la == 159:
                    self.state = 1686
                    _la = self._input.LA(1)
                    if not (_la == 146 or _la == 159):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 1687
                    localctx.ns = self.multipartIdentifier()
                pass
            elif la_ == 44:
                localctx = fugue_sqlParser.ShowViewsContext(self, localctx)
                self.enterOuterAlt(localctx, 44)
                self.state = 1690
                self.match(fugue_sqlParser.SHOW)
                self.state = 1691
                self.match(fugue_sqlParser.VIEWS)
                self.state = 1694
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 146 or _la == 159:
                    self.state = 1692
                    _la = self._input.LA(1)
                    if not (_la == 146 or _la == 159):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 1693
                    self.multipartIdentifier()
                self.state = 1700
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 178 or _la == 330:
                    self.state = 1697
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 178:
                        self.state = 1696
                        self.match(fugue_sqlParser.LIKE)
                    self.state = 1699
                    localctx.pattern = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 45:
                localctx = fugue_sqlParser.ShowPartitionsContext(self, localctx)
                self.enterOuterAlt(localctx, 45)
                self.state = 1702
                self.match(fugue_sqlParser.SHOW)
                self.state = 1703
                self.match(fugue_sqlParser.PARTITIONS)
                self.state = 1704
                self.multipartIdentifier()
                self.state = 1706
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1705
                    self.partitionSpec()
                pass
            elif la_ == 46:
                localctx = fugue_sqlParser.ShowFunctionsContext(self, localctx)
                self.enterOuterAlt(localctx, 46)
                self.state = 1708
                self.match(fugue_sqlParser.SHOW)
                self.state = 1710
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 209, self._ctx)
                if la_ == 1:
                    self.state = 1709
                    self.identifier()
                self.state = 1712
                self.match(fugue_sqlParser.FUNCTIONS)
                self.state = 1720
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 58 & ~63 == 0 and 1 << _la - 58 & -1 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -1 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -1 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1152921504606846975 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98369 != 0):
                    self.state = 1714
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 210, self._ctx)
                    if la_ == 1:
                        self.state = 1713
                        self.match(fugue_sqlParser.LIKE)
                    self.state = 1718
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                        self.state = 1716
                        self.multipartIdentifier()
                        pass
                    elif token in [330]:
                        self.state = 1717
                        localctx.pattern = self.match(fugue_sqlParser.STRING)
                        pass
                    else:
                        raise NoViableAltException(self)
                pass
            elif la_ == 47:
                localctx = fugue_sqlParser.ShowCreateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 47)
                self.state = 1722
                self.match(fugue_sqlParser.SHOW)
                self.state = 1723
                self.match(fugue_sqlParser.CREATE)
                self.state = 1724
                self.match(fugue_sqlParser.TABLE)
                self.state = 1725
                self.multipartIdentifier()
                self.state = 1728
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 68:
                    self.state = 1726
                    self.match(fugue_sqlParser.AS)
                    self.state = 1727
                    self.match(fugue_sqlParser.SERDE)
                pass
            elif la_ == 48:
                localctx = fugue_sqlParser.ShowCurrentNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 48)
                self.state = 1730
                self.match(fugue_sqlParser.SHOW)
                self.state = 1731
                self.match(fugue_sqlParser.CURRENT)
                self.state = 1732
                self.match(fugue_sqlParser.NAMESPACE)
                pass
            elif la_ == 49:
                localctx = fugue_sqlParser.DescribeFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 49)
                self.state = 1733
                _la = self._input.LA(1)
                if not (_la == 115 or _la == 116):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1734
                self.match(fugue_sqlParser.FUNCTION)
                self.state = 1736
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 214, self._ctx)
                if la_ == 1:
                    self.state = 1735
                    self.match(fugue_sqlParser.EXTENDED)
                self.state = 1738
                self.describeFuncName()
                pass
            elif la_ == 50:
                localctx = fugue_sqlParser.DescribeNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 50)
                self.state = 1739
                _la = self._input.LA(1)
                if not (_la == 115 or _la == 116):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1740
                self.theNamespace()
                self.state = 1742
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 215, self._ctx)
                if la_ == 1:
                    self.state = 1741
                    self.match(fugue_sqlParser.EXTENDED)
                self.state = 1744
                self.multipartIdentifier()
                pass
            elif la_ == 51:
                localctx = fugue_sqlParser.DescribeRelationContext(self, localctx)
                self.enterOuterAlt(localctx, 51)
                self.state = 1746
                _la = self._input.LA(1)
                if not (_la == 115 or _la == 116):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1748
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 216, self._ctx)
                if la_ == 1:
                    self.state = 1747
                    self.match(fugue_sqlParser.TABLE)
                self.state = 1751
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 217, self._ctx)
                if la_ == 1:
                    self.state = 1750
                    localctx.option = self._input.LT(1)
                    _la = self._input.LA(1)
                    if not (_la == 132 or _la == 145):
                        localctx.option = self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 1753
                self.multipartIdentifier()
                self.state = 1755
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 218, self._ctx)
                if la_ == 1:
                    self.state = 1754
                    self.partitionSpec()
                self.state = 1758
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 58 & ~63 == 0 and 1 << _la - 58 & -1 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -1 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -1 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1152921504606846975 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0):
                    self.state = 1757
                    self.describeColName()
                pass
            elif la_ == 52:
                localctx = fugue_sqlParser.DescribeQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 52)
                self.state = 1760
                _la = self._input.LA(1)
                if not (_la == 115 or _la == 116):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1762
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 228:
                    self.state = 1761
                    self.match(fugue_sqlParser.QUERY)
                self.state = 1764
                self.query()
                pass
            elif la_ == 53:
                localctx = fugue_sqlParser.CommentNamespaceContext(self, localctx)
                self.enterOuterAlt(localctx, 53)
                self.state = 1765
                self.match(fugue_sqlParser.COMMENT)
                self.state = 1766
                self.match(fugue_sqlParser.ON)
                self.state = 1767
                self.theNamespace()
                self.state = 1768
                self.multipartIdentifier()
                self.state = 1769
                self.match(fugue_sqlParser.IS)
                self.state = 1770
                localctx.comment = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 200 or _la == 330):
                    localctx.comment = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 54:
                localctx = fugue_sqlParser.CommentTableContext(self, localctx)
                self.enterOuterAlt(localctx, 54)
                self.state = 1772
                self.match(fugue_sqlParser.COMMENT)
                self.state = 1773
                self.match(fugue_sqlParser.ON)
                self.state = 1774
                self.match(fugue_sqlParser.TABLE)
                self.state = 1775
                self.multipartIdentifier()
                self.state = 1776
                self.match(fugue_sqlParser.IS)
                self.state = 1777
                localctx.comment = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 200 or _la == 330):
                    localctx.comment = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 55:
                localctx = fugue_sqlParser.RefreshTableContext(self, localctx)
                self.enterOuterAlt(localctx, 55)
                self.state = 1779
                self.match(fugue_sqlParser.REFRESH)
                self.state = 1780
                self.match(fugue_sqlParser.TABLE)
                self.state = 1781
                self.multipartIdentifier()
                pass
            elif la_ == 56:
                localctx = fugue_sqlParser.RefreshResourceContext(self, localctx)
                self.enterOuterAlt(localctx, 56)
                self.state = 1782
                self.match(fugue_sqlParser.REFRESH)
                self.state = 1790
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 222, self._ctx)
                if la_ == 1:
                    self.state = 1783
                    self.match(fugue_sqlParser.STRING)
                    pass
                elif la_ == 2:
                    self.state = 1787
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 221, self._ctx)
                    while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                        if _alt == 1 + 1:
                            self.state = 1784
                            self.matchWildcard()
                        self.state = 1789
                        self._errHandler.sync(self)
                        _alt = self._interp.adaptivePredict(self._input, 221, self._ctx)
                    pass
                pass
            elif la_ == 57:
                localctx = fugue_sqlParser.CacheTableContext(self, localctx)
                self.enterOuterAlt(localctx, 57)
                self.state = 1792
                self.match(fugue_sqlParser.CACHE)
                self.state = 1794
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 175:
                    self.state = 1793
                    self.match(fugue_sqlParser.LAZY)
                self.state = 1796
                self.match(fugue_sqlParser.TABLE)
                self.state = 1797
                self.multipartIdentifier()
                self.state = 1800
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 206:
                    self.state = 1798
                    self.match(fugue_sqlParser.OPTIONS)
                    self.state = 1799
                    localctx.options = self.tablePropertyList()
                self.state = 1806
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la - 17 & ~63 == 0 and 1 << _la - 17 & 2269395221545987 != 0 or (_la - 99 & ~63 == 0 and 1 << _la - 99 & 140737496743937 != 0) or (_la - 182 & ~63 == 0 and 1 << _la - 182 & 20266198323167361 != 0) or (_la - 252 & ~63 == 0 and 1 << _la - 252 & 73183502536802305 != 0):
                    self.state = 1803
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 68:
                        self.state = 1802
                        self.match(fugue_sqlParser.AS)
                    self.state = 1805
                    self.query()
                pass
            elif la_ == 58:
                localctx = fugue_sqlParser.UncacheTableContext(self, localctx)
                self.enterOuterAlt(localctx, 58)
                self.state = 1808
                self.match(fugue_sqlParser.UNCACHE)
                self.state = 1809
                self.match(fugue_sqlParser.TABLE)
                self.state = 1812
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 227, self._ctx)
                if la_ == 1:
                    self.state = 1810
                    self.match(fugue_sqlParser.IF)
                    self.state = 1811
                    self.match(fugue_sqlParser.EXISTS)
                self.state = 1814
                self.multipartIdentifier()
                pass
            elif la_ == 59:
                localctx = fugue_sqlParser.ClearCacheContext(self, localctx)
                self.enterOuterAlt(localctx, 59)
                self.state = 1815
                self.match(fugue_sqlParser.CLEAR)
                self.state = 1816
                self.match(fugue_sqlParser.CACHE)
                pass
            elif la_ == 60:
                localctx = fugue_sqlParser.LoadDataContext(self, localctx)
                self.enterOuterAlt(localctx, 60)
                self.state = 1817
                self.match(fugue_sqlParser.LOAD)
                self.state = 1818
                self.match(fugue_sqlParser.DATA)
                self.state = 1820
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 183:
                    self.state = 1819
                    self.match(fugue_sqlParser.LOCAL)
                self.state = 1822
                self.match(fugue_sqlParser.INPATH)
                self.state = 1823
                localctx.path = self.match(fugue_sqlParser.STRING)
                self.state = 1825
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 215:
                    self.state = 1824
                    self.match(fugue_sqlParser.OVERWRITE)
                self.state = 1827
                self.match(fugue_sqlParser.INTO)
                self.state = 1828
                self.match(fugue_sqlParser.TABLE)
                self.state = 1829
                self.multipartIdentifier()
                self.state = 1831
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1830
                    self.partitionSpec()
                pass
            elif la_ == 61:
                localctx = fugue_sqlParser.TruncateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 61)
                self.state = 1833
                self.match(fugue_sqlParser.TRUNCATE)
                self.state = 1834
                self.match(fugue_sqlParser.TABLE)
                self.state = 1835
                self.multipartIdentifier()
                self.state = 1837
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 1836
                    self.partitionSpec()
                pass
            elif la_ == 62:
                localctx = fugue_sqlParser.RepairTableContext(self, localctx)
                self.enterOuterAlt(localctx, 62)
                self.state = 1839
                self.match(fugue_sqlParser.MSCK)
                self.state = 1840
                self.match(fugue_sqlParser.REPAIR)
                self.state = 1841
                self.match(fugue_sqlParser.TABLE)
                self.state = 1842
                self.multipartIdentifier()
                pass
            elif la_ == 63:
                localctx = fugue_sqlParser.ManageResourceContext(self, localctx)
                self.enterOuterAlt(localctx, 63)
                self.state = 1843
                localctx.op = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 58 or _la == 181):
                    localctx.op = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 1844
                self.identifier()
                self.state = 1852
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 233, self._ctx)
                if la_ == 1:
                    self.state = 1845
                    self.match(fugue_sqlParser.STRING)
                    pass
                elif la_ == 2:
                    self.state = 1849
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 232, self._ctx)
                    while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                        if _alt == 1 + 1:
                            self.state = 1846
                            self.matchWildcard()
                        self.state = 1851
                        self._errHandler.sync(self)
                        _alt = self._interp.adaptivePredict(self._input, 232, self._ctx)
                    pass
                pass
            elif la_ == 64:
                localctx = fugue_sqlParser.FailNativeCommandContext(self, localctx)
                self.enterOuterAlt(localctx, 64)
                self.state = 1854
                self.match(fugue_sqlParser.SET)
                self.state = 1855
                self.match(fugue_sqlParser.ROLE)
                self.state = 1859
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 234, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 1856
                        self.matchWildcard()
                    self.state = 1861
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 234, self._ctx)
                pass
            elif la_ == 65:
                localctx = fugue_sqlParser.SetConfigurationContext(self, localctx)
                self.enterOuterAlt(localctx, 65)
                self.state = 1862
                self.match(fugue_sqlParser.SET)
                self.state = 1866
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 235, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 1863
                        self.matchWildcard()
                    self.state = 1868
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 235, self._ctx)
                pass
            elif la_ == 66:
                localctx = fugue_sqlParser.ResetConfigurationContext(self, localctx)
                self.enterOuterAlt(localctx, 66)
                self.state = 1869
                self.match(fugue_sqlParser.RESET)
                pass
            elif la_ == 67:
                localctx = fugue_sqlParser.FailNativeCommandContext(self, localctx)
                self.enterOuterAlt(localctx, 67)
                self.state = 1870
                self.unsupportedHiveNativeCommands()
                self.state = 1874
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 236, self._ctx)
                while _alt != 1 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1 + 1:
                        self.state = 1871
                        self.matchWildcard()
                    self.state = 1876
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 236, self._ctx)
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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def ROLE(self):
            return self.getToken(fugue_sqlParser.ROLE, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def GRANT(self):
            return self.getToken(fugue_sqlParser.GRANT, 0)

        def REVOKE(self):
            return self.getToken(fugue_sqlParser.REVOKE, 0)

        def SHOW(self):
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def PRINCIPALS(self):
            return self.getToken(fugue_sqlParser.PRINCIPALS, 0)

        def ROLES(self):
            return self.getToken(fugue_sqlParser.ROLES, 0)

        def CURRENT(self):
            return self.getToken(fugue_sqlParser.CURRENT, 0)

        def EXPORT(self):
            return self.getToken(fugue_sqlParser.EXPORT, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def IMPORT(self):
            return self.getToken(fugue_sqlParser.IMPORT, 0)

        def COMPACTIONS(self):
            return self.getToken(fugue_sqlParser.COMPACTIONS, 0)

        def TRANSACTIONS(self):
            return self.getToken(fugue_sqlParser.TRANSACTIONS, 0)

        def INDEXES(self):
            return self.getToken(fugue_sqlParser.INDEXES, 0)

        def LOCKS(self):
            return self.getToken(fugue_sqlParser.LOCKS, 0)

        def INDEX(self):
            return self.getToken(fugue_sqlParser.INDEX, 0)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def LOCK(self):
            return self.getToken(fugue_sqlParser.LOCK, 0)

        def DATABASE(self):
            return self.getToken(fugue_sqlParser.DATABASE, 0)

        def UNLOCK(self):
            return self.getToken(fugue_sqlParser.UNLOCK, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def MACRO(self):
            return self.getToken(fugue_sqlParser.MACRO, 0)

        def tableIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableIdentifierContext, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def CLUSTERED(self):
            return self.getToken(fugue_sqlParser.CLUSTERED, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def SORTED(self):
            return self.getToken(fugue_sqlParser.SORTED, 0)

        def SKEWED(self):
            return self.getToken(fugue_sqlParser.SKEWED, 0)

        def STORED(self):
            return self.getToken(fugue_sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def DIRECTORIES(self):
            return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def LOCATION(self):
            return self.getToken(fugue_sqlParser.LOCATION, 0)

        def EXCHANGE(self):
            return self.getToken(fugue_sqlParser.EXCHANGE, 0)

        def PARTITION(self):
            return self.getToken(fugue_sqlParser.PARTITION, 0)

        def ARCHIVE(self):
            return self.getToken(fugue_sqlParser.ARCHIVE, 0)

        def UNARCHIVE(self):
            return self.getToken(fugue_sqlParser.UNARCHIVE, 0)

        def TOUCH(self):
            return self.getToken(fugue_sqlParser.TOUCH, 0)

        def COMPACT(self):
            return self.getToken(fugue_sqlParser.COMPACT, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def CONCATENATE(self):
            return self.getToken(fugue_sqlParser.CONCATENATE, 0)

        def FILEFORMAT(self):
            return self.getToken(fugue_sqlParser.FILEFORMAT, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def START(self):
            return self.getToken(fugue_sqlParser.START, 0)

        def TRANSACTION(self):
            return self.getToken(fugue_sqlParser.TRANSACTION, 0)

        def COMMIT(self):
            return self.getToken(fugue_sqlParser.COMMIT, 0)

        def ROLLBACK(self):
            return self.getToken(fugue_sqlParser.ROLLBACK, 0)

        def DFS(self):
            return self.getToken(fugue_sqlParser.DFS, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_unsupportedHiveNativeCommands

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnsupportedHiveNativeCommands'):
                return visitor.visitUnsupportedHiveNativeCommands(self)
            else:
                return visitor.visitChildren(self)

    def unsupportedHiveNativeCommands(self):
        localctx = fugue_sqlParser.UnsupportedHiveNativeCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 168, self.RULE_unsupportedHiveNativeCommands)
        self._la = 0
        try:
            self.state = 2047
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 245, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 1879
                localctx.kw1 = self.match(fugue_sqlParser.CREATE)
                self.state = 1880
                localctx.kw2 = self.match(fugue_sqlParser.ROLE)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 1881
                localctx.kw1 = self.match(fugue_sqlParser.DROP)
                self.state = 1882
                localctx.kw2 = self.match(fugue_sqlParser.ROLE)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 1883
                localctx.kw1 = self.match(fugue_sqlParser.GRANT)
                self.state = 1885
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 238, self._ctx)
                if la_ == 1:
                    self.state = 1884
                    localctx.kw2 = self.match(fugue_sqlParser.ROLE)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 1887
                localctx.kw1 = self.match(fugue_sqlParser.REVOKE)
                self.state = 1889
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 239, self._ctx)
                if la_ == 1:
                    self.state = 1888
                    localctx.kw2 = self.match(fugue_sqlParser.ROLE)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 1891
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1892
                localctx.kw2 = self.match(fugue_sqlParser.GRANT)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 1893
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1894
                localctx.kw2 = self.match(fugue_sqlParser.ROLE)
                self.state = 1896
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 240, self._ctx)
                if la_ == 1:
                    self.state = 1895
                    localctx.kw3 = self.match(fugue_sqlParser.GRANT)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 1898
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1899
                localctx.kw2 = self.match(fugue_sqlParser.PRINCIPALS)
                pass
            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 1900
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1901
                localctx.kw2 = self.match(fugue_sqlParser.ROLES)
                pass
            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 1902
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1903
                localctx.kw2 = self.match(fugue_sqlParser.CURRENT)
                self.state = 1904
                localctx.kw3 = self.match(fugue_sqlParser.ROLES)
                pass
            elif la_ == 10:
                self.enterOuterAlt(localctx, 10)
                self.state = 1905
                localctx.kw1 = self.match(fugue_sqlParser.EXPORT)
                self.state = 1906
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                pass
            elif la_ == 11:
                self.enterOuterAlt(localctx, 11)
                self.state = 1907
                localctx.kw1 = self.match(fugue_sqlParser.IMPORT)
                self.state = 1908
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                pass
            elif la_ == 12:
                self.enterOuterAlt(localctx, 12)
                self.state = 1909
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1910
                localctx.kw2 = self.match(fugue_sqlParser.COMPACTIONS)
                pass
            elif la_ == 13:
                self.enterOuterAlt(localctx, 13)
                self.state = 1911
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1912
                localctx.kw2 = self.match(fugue_sqlParser.CREATE)
                self.state = 1913
                localctx.kw3 = self.match(fugue_sqlParser.TABLE)
                pass
            elif la_ == 14:
                self.enterOuterAlt(localctx, 14)
                self.state = 1914
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1915
                localctx.kw2 = self.match(fugue_sqlParser.TRANSACTIONS)
                pass
            elif la_ == 15:
                self.enterOuterAlt(localctx, 15)
                self.state = 1916
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1917
                localctx.kw2 = self.match(fugue_sqlParser.INDEXES)
                pass
            elif la_ == 16:
                self.enterOuterAlt(localctx, 16)
                self.state = 1918
                localctx.kw1 = self.match(fugue_sqlParser.SHOW)
                self.state = 1919
                localctx.kw2 = self.match(fugue_sqlParser.LOCKS)
                pass
            elif la_ == 17:
                self.enterOuterAlt(localctx, 17)
                self.state = 1920
                localctx.kw1 = self.match(fugue_sqlParser.CREATE)
                self.state = 1921
                localctx.kw2 = self.match(fugue_sqlParser.INDEX)
                pass
            elif la_ == 18:
                self.enterOuterAlt(localctx, 18)
                self.state = 1922
                localctx.kw1 = self.match(fugue_sqlParser.DROP)
                self.state = 1923
                localctx.kw2 = self.match(fugue_sqlParser.INDEX)
                pass
            elif la_ == 19:
                self.enterOuterAlt(localctx, 19)
                self.state = 1924
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1925
                localctx.kw2 = self.match(fugue_sqlParser.INDEX)
                pass
            elif la_ == 20:
                self.enterOuterAlt(localctx, 20)
                self.state = 1926
                localctx.kw1 = self.match(fugue_sqlParser.LOCK)
                self.state = 1927
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                pass
            elif la_ == 21:
                self.enterOuterAlt(localctx, 21)
                self.state = 1928
                localctx.kw1 = self.match(fugue_sqlParser.LOCK)
                self.state = 1929
                localctx.kw2 = self.match(fugue_sqlParser.DATABASE)
                pass
            elif la_ == 22:
                self.enterOuterAlt(localctx, 22)
                self.state = 1930
                localctx.kw1 = self.match(fugue_sqlParser.UNLOCK)
                self.state = 1931
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                pass
            elif la_ == 23:
                self.enterOuterAlt(localctx, 23)
                self.state = 1932
                localctx.kw1 = self.match(fugue_sqlParser.UNLOCK)
                self.state = 1933
                localctx.kw2 = self.match(fugue_sqlParser.DATABASE)
                pass
            elif la_ == 24:
                self.enterOuterAlt(localctx, 24)
                self.state = 1934
                localctx.kw1 = self.match(fugue_sqlParser.CREATE)
                self.state = 1935
                localctx.kw2 = self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1936
                localctx.kw3 = self.match(fugue_sqlParser.MACRO)
                pass
            elif la_ == 25:
                self.enterOuterAlt(localctx, 25)
                self.state = 1937
                localctx.kw1 = self.match(fugue_sqlParser.DROP)
                self.state = 1938
                localctx.kw2 = self.match(fugue_sqlParser.TEMPORARY)
                self.state = 1939
                localctx.kw3 = self.match(fugue_sqlParser.MACRO)
                pass
            elif la_ == 26:
                self.enterOuterAlt(localctx, 26)
                self.state = 1940
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1941
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1942
                self.tableIdentifier()
                self.state = 1943
                localctx.kw3 = self.match(fugue_sqlParser.NOT)
                self.state = 1944
                localctx.kw4 = self.match(fugue_sqlParser.CLUSTERED)
                pass
            elif la_ == 27:
                self.enterOuterAlt(localctx, 27)
                self.state = 1946
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1947
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1948
                self.tableIdentifier()
                self.state = 1949
                localctx.kw3 = self.match(fugue_sqlParser.CLUSTERED)
                self.state = 1950
                localctx.kw4 = self.match(fugue_sqlParser.BY)
                pass
            elif la_ == 28:
                self.enterOuterAlt(localctx, 28)
                self.state = 1952
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1953
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1954
                self.tableIdentifier()
                self.state = 1955
                localctx.kw3 = self.match(fugue_sqlParser.NOT)
                self.state = 1956
                localctx.kw4 = self.match(fugue_sqlParser.SORTED)
                pass
            elif la_ == 29:
                self.enterOuterAlt(localctx, 29)
                self.state = 1958
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1959
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1960
                self.tableIdentifier()
                self.state = 1961
                localctx.kw3 = self.match(fugue_sqlParser.SKEWED)
                self.state = 1962
                localctx.kw4 = self.match(fugue_sqlParser.BY)
                pass
            elif la_ == 30:
                self.enterOuterAlt(localctx, 30)
                self.state = 1964
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1965
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1966
                self.tableIdentifier()
                self.state = 1967
                localctx.kw3 = self.match(fugue_sqlParser.NOT)
                self.state = 1968
                localctx.kw4 = self.match(fugue_sqlParser.SKEWED)
                pass
            elif la_ == 31:
                self.enterOuterAlt(localctx, 31)
                self.state = 1970
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1971
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1972
                self.tableIdentifier()
                self.state = 1973
                localctx.kw3 = self.match(fugue_sqlParser.NOT)
                self.state = 1974
                localctx.kw4 = self.match(fugue_sqlParser.STORED)
                self.state = 1975
                localctx.kw5 = self.match(fugue_sqlParser.AS)
                self.state = 1976
                localctx.kw6 = self.match(fugue_sqlParser.DIRECTORIES)
                pass
            elif la_ == 32:
                self.enterOuterAlt(localctx, 32)
                self.state = 1978
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1979
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1980
                self.tableIdentifier()
                self.state = 1981
                localctx.kw3 = self.match(fugue_sqlParser.SET)
                self.state = 1982
                localctx.kw4 = self.match(fugue_sqlParser.SKEWED)
                self.state = 1983
                localctx.kw5 = self.match(fugue_sqlParser.LOCATION)
                pass
            elif la_ == 33:
                self.enterOuterAlt(localctx, 33)
                self.state = 1985
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1986
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1987
                self.tableIdentifier()
                self.state = 1988
                localctx.kw3 = self.match(fugue_sqlParser.EXCHANGE)
                self.state = 1989
                localctx.kw4 = self.match(fugue_sqlParser.PARTITION)
                pass
            elif la_ == 34:
                self.enterOuterAlt(localctx, 34)
                self.state = 1991
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1992
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1993
                self.tableIdentifier()
                self.state = 1994
                localctx.kw3 = self.match(fugue_sqlParser.ARCHIVE)
                self.state = 1995
                localctx.kw4 = self.match(fugue_sqlParser.PARTITION)
                pass
            elif la_ == 35:
                self.enterOuterAlt(localctx, 35)
                self.state = 1997
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 1998
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 1999
                self.tableIdentifier()
                self.state = 2000
                localctx.kw3 = self.match(fugue_sqlParser.UNARCHIVE)
                self.state = 2001
                localctx.kw4 = self.match(fugue_sqlParser.PARTITION)
                pass
            elif la_ == 36:
                self.enterOuterAlt(localctx, 36)
                self.state = 2003
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 2004
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 2005
                self.tableIdentifier()
                self.state = 2006
                localctx.kw3 = self.match(fugue_sqlParser.TOUCH)
                pass
            elif la_ == 37:
                self.enterOuterAlt(localctx, 37)
                self.state = 2008
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 2009
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 2010
                self.tableIdentifier()
                self.state = 2012
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2011
                    self.partitionSpec()
                self.state = 2014
                localctx.kw3 = self.match(fugue_sqlParser.COMPACT)
                pass
            elif la_ == 38:
                self.enterOuterAlt(localctx, 38)
                self.state = 2016
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 2017
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 2018
                self.tableIdentifier()
                self.state = 2020
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2019
                    self.partitionSpec()
                self.state = 2022
                localctx.kw3 = self.match(fugue_sqlParser.CONCATENATE)
                pass
            elif la_ == 39:
                self.enterOuterAlt(localctx, 39)
                self.state = 2024
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 2025
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 2026
                self.tableIdentifier()
                self.state = 2028
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2027
                    self.partitionSpec()
                self.state = 2030
                localctx.kw3 = self.match(fugue_sqlParser.SET)
                self.state = 2031
                localctx.kw4 = self.match(fugue_sqlParser.FILEFORMAT)
                pass
            elif la_ == 40:
                self.enterOuterAlt(localctx, 40)
                self.state = 2033
                localctx.kw1 = self.match(fugue_sqlParser.ALTER)
                self.state = 2034
                localctx.kw2 = self.match(fugue_sqlParser.TABLE)
                self.state = 2035
                self.tableIdentifier()
                self.state = 2037
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2036
                    self.partitionSpec()
                self.state = 2039
                localctx.kw3 = self.match(fugue_sqlParser.REPLACE)
                self.state = 2040
                localctx.kw4 = self.match(fugue_sqlParser.COLUMNS)
                pass
            elif la_ == 41:
                self.enterOuterAlt(localctx, 41)
                self.state = 2042
                localctx.kw1 = self.match(fugue_sqlParser.START)
                self.state = 2043
                localctx.kw2 = self.match(fugue_sqlParser.TRANSACTION)
                pass
            elif la_ == 42:
                self.enterOuterAlt(localctx, 42)
                self.state = 2044
                localctx.kw1 = self.match(fugue_sqlParser.COMMIT)
                pass
            elif la_ == 43:
                self.enterOuterAlt(localctx, 43)
                self.state = 2045
                localctx.kw1 = self.match(fugue_sqlParser.ROLLBACK)
                pass
            elif la_ == 44:
                self.enterOuterAlt(localctx, 44)
                self.state = 2046
                localctx.kw1 = self.match(fugue_sqlParser.DFS)
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
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def EXTERNAL(self):
            return self.getToken(fugue_sqlParser.EXTERNAL, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_createTableHeader

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTableHeader'):
                return visitor.visitCreateTableHeader(self)
            else:
                return visitor.visitChildren(self)

    def createTableHeader(self):
        localctx = fugue_sqlParser.CreateTableHeaderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 170, self.RULE_createTableHeader)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2049
            self.match(fugue_sqlParser.CREATE)
            self.state = 2051
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 277:
                self.state = 2050
                self.match(fugue_sqlParser.TEMPORARY)
            self.state = 2054
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 133:
                self.state = 2053
                self.match(fugue_sqlParser.EXTERNAL)
            self.state = 2056
            self.match(fugue_sqlParser.TABLE)
            self.state = 2060
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 248, self._ctx)
            if la_ == 1:
                self.state = 2057
                self.match(fugue_sqlParser.IF)
                self.state = 2058
                self.match(fugue_sqlParser.NOT)
                self.state = 2059
                self.match(fugue_sqlParser.EXISTS)
            self.state = 2062
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
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def CREATE(self):
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_replaceTableHeader

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitReplaceTableHeader'):
                return visitor.visitReplaceTableHeader(self)
            else:
                return visitor.visitChildren(self)

    def replaceTableHeader(self):
        localctx = fugue_sqlParser.ReplaceTableHeaderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 172, self.RULE_replaceTableHeader)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2066
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 99:
                self.state = 2064
                self.match(fugue_sqlParser.CREATE)
                self.state = 2065
                self.match(fugue_sqlParser.OR)
            self.state = 2068
            self.match(fugue_sqlParser.REPLACE)
            self.state = 2069
            self.match(fugue_sqlParser.TABLE)
            self.state = 2070
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
            return self.getToken(fugue_sqlParser.CLUSTERED, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def INTO(self):
            return self.getToken(fugue_sqlParser.INTO, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def BUCKETS(self):
            return self.getToken(fugue_sqlParser.BUCKETS, 0)

        def SORTED(self):
            return self.getToken(fugue_sqlParser.SORTED, 0)

        def orderedIdentifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.OrderedIdentifierListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_bucketSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBucketSpec'):
                return visitor.visitBucketSpec(self)
            else:
                return visitor.visitChildren(self)

    def bucketSpec(self):
        localctx = fugue_sqlParser.BucketSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 174, self.RULE_bucketSpec)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2072
            self.match(fugue_sqlParser.CLUSTERED)
            self.state = 2073
            self.match(fugue_sqlParser.BY)
            self.state = 2074
            self.identifierList()
            self.state = 2078
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 265:
                self.state = 2075
                self.match(fugue_sqlParser.SORTED)
                self.state = 2076
                self.match(fugue_sqlParser.BY)
                self.state = 2077
                self.orderedIdentifierList()
            self.state = 2080
            self.match(fugue_sqlParser.INTO)
            self.state = 2081
            self.match(fugue_sqlParser.INTEGER_VALUE)
            self.state = 2082
            self.match(fugue_sqlParser.BUCKETS)
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
            return self.getToken(fugue_sqlParser.SKEWED, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def constantList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ConstantListContext, 0)

        def nestedConstantList(self):
            return self.getTypedRuleContext(fugue_sqlParser.NestedConstantListContext, 0)

        def STORED(self):
            return self.getToken(fugue_sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def DIRECTORIES(self):
            return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_skewSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSkewSpec'):
                return visitor.visitSkewSpec(self)
            else:
                return visitor.visitChildren(self)

    def skewSpec(self):
        localctx = fugue_sqlParser.SkewSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 176, self.RULE_skewSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2084
            self.match(fugue_sqlParser.SKEWED)
            self.state = 2085
            self.match(fugue_sqlParser.BY)
            self.state = 2086
            self.identifierList()
            self.state = 2087
            self.match(fugue_sqlParser.ON)
            self.state = 2090
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 251, self._ctx)
            if la_ == 1:
                self.state = 2088
                self.constantList()
                pass
            elif la_ == 2:
                self.state = 2089
                self.nestedConstantList()
                pass
            self.state = 2095
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 252, self._ctx)
            if la_ == 1:
                self.state = 2092
                self.match(fugue_sqlParser.STORED)
                self.state = 2093
                self.match(fugue_sqlParser.AS)
                self.state = 2094
                self.match(fugue_sqlParser.DIRECTORIES)
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
            return self.getToken(fugue_sqlParser.LOCATION, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_locationSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLocationSpec'):
                return visitor.visitLocationSpec(self)
            else:
                return visitor.visitChildren(self)

    def locationSpec(self):
        localctx = fugue_sqlParser.LocationSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 178, self.RULE_locationSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2097
            self.match(fugue_sqlParser.LOCATION)
            self.state = 2098
            self.match(fugue_sqlParser.STRING)
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
            return self.getToken(fugue_sqlParser.COMMENT, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_commentSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCommentSpec'):
                return visitor.visitCommentSpec(self)
            else:
                return visitor.visitChildren(self)

    def commentSpec(self):
        localctx = fugue_sqlParser.CommentSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 180, self.RULE_commentSpec)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2100
            self.match(fugue_sqlParser.COMMENT)
            self.state = 2101
            self.match(fugue_sqlParser.STRING)
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
            return self.getTypedRuleContext(fugue_sqlParser.QueryTermContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryOrganizationContext, 0)

        def fugueSqlEngine(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueSqlEngineContext, 0)

        def ctes(self):
            return self.getTypedRuleContext(fugue_sqlParser.CtesContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_query

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQuery'):
                return visitor.visitQuery(self)
            else:
                return visitor.visitChildren(self)

    def query(self):
        localctx = fugue_sqlParser.QueryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 182, self.RULE_query)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2104
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 47:
                self.state = 2103
                self.fugueSqlEngine()
            self.state = 2107
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 308:
                self.state = 2106
                self.ctes()
            self.state = 2109
            self.queryTerm(0)
            self.state = 2110
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
            return fugue_sqlParser.RULE_insertInto

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class InsertOverwriteHiveDirContext(InsertIntoContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.path = None
            self.copyFrom(ctx)

        def INSERT(self):
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def DIRECTORY(self):
            return self.getToken(fugue_sqlParser.DIRECTORY, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def rowFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, 0)

        def createFileFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, 0)

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
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def DIRECTORY(self):
            return self.getToken(fugue_sqlParser.DIRECTORY, 0)

        def tableProvider(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableProviderContext, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def OPTIONS(self):
            return self.getToken(fugue_sqlParser.OPTIONS, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

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
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

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
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def INTO(self):
            return self.getToken(fugue_sqlParser.INTO, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def partitionSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInsertIntoTable'):
                return visitor.visitInsertIntoTable(self)
            else:
                return visitor.visitChildren(self)

    def insertInto(self):
        localctx = fugue_sqlParser.InsertIntoContext(self, self._ctx, self.state)
        self.enterRule(localctx, 184, self.RULE_insertInto)
        self._la = 0
        try:
            self.state = 2167
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 267, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.InsertOverwriteTableContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2112
                self.match(fugue_sqlParser.INSERT)
                self.state = 2113
                self.match(fugue_sqlParser.OVERWRITE)
                self.state = 2115
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 255, self._ctx)
                if la_ == 1:
                    self.state = 2114
                    self.match(fugue_sqlParser.TABLE)
                self.state = 2117
                self.multipartIdentifier()
                self.state = 2124
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2118
                    self.partitionSpec()
                    self.state = 2122
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 156:
                        self.state = 2119
                        self.match(fugue_sqlParser.IF)
                        self.state = 2120
                        self.match(fugue_sqlParser.NOT)
                        self.state = 2121
                        self.match(fugue_sqlParser.EXISTS)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.InsertIntoTableContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2126
                self.match(fugue_sqlParser.INSERT)
                self.state = 2127
                self.match(fugue_sqlParser.INTO)
                self.state = 2129
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 258, self._ctx)
                if la_ == 1:
                    self.state = 2128
                    self.match(fugue_sqlParser.TABLE)
                self.state = 2131
                self.multipartIdentifier()
                self.state = 2133
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 216:
                    self.state = 2132
                    self.partitionSpec()
                self.state = 2138
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 156:
                    self.state = 2135
                    self.match(fugue_sqlParser.IF)
                    self.state = 2136
                    self.match(fugue_sqlParser.NOT)
                    self.state = 2137
                    self.match(fugue_sqlParser.EXISTS)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.InsertOverwriteHiveDirContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2140
                self.match(fugue_sqlParser.INSERT)
                self.state = 2141
                self.match(fugue_sqlParser.OVERWRITE)
                self.state = 2143
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 183:
                    self.state = 2142
                    self.match(fugue_sqlParser.LOCAL)
                self.state = 2145
                self.match(fugue_sqlParser.DIRECTORY)
                self.state = 2146
                localctx.path = self.match(fugue_sqlParser.STRING)
                self.state = 2148
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 248:
                    self.state = 2147
                    self.rowFormat()
                self.state = 2151
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 268:
                    self.state = 2150
                    self.createFileFormat()
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.InsertOverwriteDirContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2153
                self.match(fugue_sqlParser.INSERT)
                self.state = 2154
                self.match(fugue_sqlParser.OVERWRITE)
                self.state = 2156
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 183:
                    self.state = 2155
                    self.match(fugue_sqlParser.LOCAL)
                self.state = 2158
                self.match(fugue_sqlParser.DIRECTORY)
                self.state = 2160
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 330:
                    self.state = 2159
                    localctx.path = self.match(fugue_sqlParser.STRING)
                self.state = 2162
                self.tableProvider()
                self.state = 2165
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 206:
                    self.state = 2163
                    self.match(fugue_sqlParser.OPTIONS)
                    self.state = 2164
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
            return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

        def locationSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_partitionSpecLocation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionSpecLocation'):
                return visitor.visitPartitionSpecLocation(self)
            else:
                return visitor.visitChildren(self)

    def partitionSpecLocation(self):
        localctx = fugue_sqlParser.PartitionSpecLocationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 186, self.RULE_partitionSpecLocation)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2169
            self.partitionSpec()
            self.state = 2171
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 184:
                self.state = 2170
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
            return self.getToken(fugue_sqlParser.PARTITION, 0)

        def partitionVal(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PartitionValContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PartitionValContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_partitionSpec

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionSpec'):
                return visitor.visitPartitionSpec(self)
            else:
                return visitor.visitChildren(self)

    def partitionSpec(self):
        localctx = fugue_sqlParser.PartitionSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 188, self.RULE_partitionSpec)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2173
            self.match(fugue_sqlParser.PARTITION)
            self.state = 2174
            self.match(fugue_sqlParser.T__4)
            self.state = 2175
            self.partitionVal()
            self.state = 2180
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2176
                self.match(fugue_sqlParser.T__1)
                self.state = 2177
                self.partitionVal()
                self.state = 2182
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2183
            self.match(fugue_sqlParser.T__5)
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def constant(self):
            return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_partitionVal

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPartitionVal'):
                return visitor.visitPartitionVal(self)
            else:
                return visitor.visitChildren(self)

    def partitionVal(self):
        localctx = fugue_sqlParser.PartitionValContext(self, self._ctx, self.state)
        self.enterRule(localctx, 190, self.RULE_partitionVal)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2185
            self.identifier()
            self.state = 2188
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 310:
                self.state = 2186
                self.match(fugue_sqlParser.EQUAL)
                self.state = 2187
                self.constant()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TheNamespaceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAMESPACE(self):
            return self.getToken(fugue_sqlParser.NAMESPACE, 0)

        def DATABASE(self):
            return self.getToken(fugue_sqlParser.DATABASE, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_theNamespace

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTheNamespace'):
                return visitor.visitTheNamespace(self)
            else:
                return visitor.visitChildren(self)

    def theNamespace(self):
        localctx = fugue_sqlParser.TheNamespaceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 192, self.RULE_theNamespace)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2190
            _la = self._input.LA(1)
            if not (_la == 108 or _la == 195 or _la == 250):
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
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def comparisonOperator(self):
            return self.getTypedRuleContext(fugue_sqlParser.ComparisonOperatorContext, 0)

        def arithmeticOperator(self):
            return self.getTypedRuleContext(fugue_sqlParser.ArithmeticOperatorContext, 0)

        def predicateOperator(self):
            return self.getTypedRuleContext(fugue_sqlParser.PredicateOperatorContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_describeFuncName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeFuncName'):
                return visitor.visitDescribeFuncName(self)
            else:
                return visitor.visitChildren(self)

    def describeFuncName(self):
        localctx = fugue_sqlParser.DescribeFuncNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 194, self.RULE_describeFuncName)
        try:
            self.state = 2197
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 271, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2192
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2193
                self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2194
                self.comparisonOperator()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2195
                self.arithmeticOperator()
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 2196
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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_describeColName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitDescribeColName'):
                return visitor.visitDescribeColName(self)
            else:
                return visitor.visitChildren(self)

    def describeColName(self):
        localctx = fugue_sqlParser.DescribeColNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 196, self.RULE_describeColName)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2199
            localctx._identifier = self.identifier()
            localctx.nameParts.append(localctx._identifier)
            self.state = 2204
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 7:
                self.state = 2200
                self.match(fugue_sqlParser.T__6)
                self.state = 2201
                localctx._identifier = self.identifier()
                localctx.nameParts.append(localctx._identifier)
                self.state = 2206
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
            return self.getToken(fugue_sqlParser.WITH, 0)

        def namedQuery(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.NamedQueryContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NamedQueryContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_ctes

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCtes'):
                return visitor.visitCtes(self)
            else:
                return visitor.visitChildren(self)

    def ctes(self):
        localctx = fugue_sqlParser.CtesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 198, self.RULE_ctes)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2207
            self.match(fugue_sqlParser.WITH)
            self.state = 2208
            self.namedQuery()
            self.state = 2213
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2209
                self.match(fugue_sqlParser.T__1)
                self.state = 2210
                self.namedQuery()
                self.state = 2215
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
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_namedQuery

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedQuery'):
                return visitor.visitNamedQuery(self)
            else:
                return visitor.visitChildren(self)

    def namedQuery(self):
        localctx = fugue_sqlParser.NamedQueryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 200, self.RULE_namedQuery)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2216
            localctx.name = self.errorCapturingIdentifier()
            self.state = 2218
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 274, self._ctx)
            if la_ == 1:
                self.state = 2217
                localctx.columnAliases = self.identifierList()
            self.state = 2221
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 68:
                self.state = 2220
                self.match(fugue_sqlParser.AS)
            self.state = 2223
            self.match(fugue_sqlParser.T__4)
            self.state = 2224
            self.query()
            self.state = 2225
            self.match(fugue_sqlParser.T__5)
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
            return self.getToken(fugue_sqlParser.USING, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tableProvider

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableProvider'):
                return visitor.visitTableProvider(self)
            else:
                return visitor.visitChildren(self)

    def tableProvider(self):
        localctx = fugue_sqlParser.TableProviderContext(self, self._ctx, self.state)
        self.enterRule(localctx, 202, self.RULE_tableProvider)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2227
            self.match(fugue_sqlParser.USING)
            self.state = 2228
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
                return self.getTypedRuleContexts(fugue_sqlParser.BucketSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.BucketSpecContext, i)

        def locationSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

        def commentSpec(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

        def OPTIONS(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.OPTIONS)
            else:
                return self.getToken(fugue_sqlParser.OPTIONS, i)

        def PARTITIONED(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.PARTITIONED)
            else:
                return self.getToken(fugue_sqlParser.PARTITIONED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def TBLPROPERTIES(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
            else:
                return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

        def tablePropertyList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

        def transformList(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TransformListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TransformListContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_createTableClauses

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateTableClauses'):
                return visitor.visitCreateTableClauses(self)
            else:
                return visitor.visitChildren(self)

    def createTableClauses(self):
        localctx = fugue_sqlParser.CreateTableClausesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 204, self.RULE_createTableClauses)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2242
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 85 or _la == 91 or (_la - 184 & ~63 == 0 and 1 << _la - 184 & 8594128897 != 0) or (_la == 276):
                self.state = 2240
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [206]:
                    self.state = 2230
                    self.match(fugue_sqlParser.OPTIONS)
                    self.state = 2231
                    localctx.options = self.tablePropertyList()
                    pass
                elif token in [217]:
                    self.state = 2232
                    self.match(fugue_sqlParser.PARTITIONED)
                    self.state = 2233
                    self.match(fugue_sqlParser.BY)
                    self.state = 2234
                    localctx.partitioning = self.transformList()
                    pass
                elif token in [85]:
                    self.state = 2235
                    self.bucketSpec()
                    pass
                elif token in [184]:
                    self.state = 2236
                    self.locationSpec()
                    pass
                elif token in [91]:
                    self.state = 2237
                    self.commentSpec()
                    pass
                elif token in [276]:
                    self.state = 2238
                    self.match(fugue_sqlParser.TBLPROPERTIES)
                    self.state = 2239
                    localctx.tableProps = self.tablePropertyList()
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 2244
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
                return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TablePropertyContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tablePropertyList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyList'):
                return visitor.visitTablePropertyList(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyList(self):
        localctx = fugue_sqlParser.TablePropertyListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 206, self.RULE_tablePropertyList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2245
            self.match(fugue_sqlParser.T__4)
            self.state = 2246
            self.tableProperty()
            self.state = 2251
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2247
                self.match(fugue_sqlParser.T__1)
                self.state = 2248
                self.tableProperty()
                self.state = 2253
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2254
            self.match(fugue_sqlParser.T__5)
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
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyKeyContext, 0)

        def tablePropertyValue(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyValueContext, 0)

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tableProperty

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableProperty'):
                return visitor.visitTableProperty(self)
            else:
                return visitor.visitChildren(self)

    def tableProperty(self):
        localctx = fugue_sqlParser.TablePropertyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 208, self.RULE_tableProperty)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2256
            localctx.key = self.tablePropertyKey()
            self.state = 2261
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 135 or (_la - 287 & ~63 == 0 and 1 << _la - 287 & 712483543187457 != 0):
                self.state = 2258
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 310:
                    self.state = 2257
                    self.match(fugue_sqlParser.EQUAL)
                self.state = 2260
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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tablePropertyKey

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyKey'):
                return visitor.visitTablePropertyKey(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyKey(self):
        localctx = fugue_sqlParser.TablePropertyKeyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 210, self.RULE_tablePropertyKey)
        self._la = 0
        try:
            self.state = 2272
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2263
                self.identifier()
                self.state = 2268
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 7:
                    self.state = 2264
                    self.match(fugue_sqlParser.T__6)
                    self.state = 2265
                    self.identifier()
                    self.state = 2270
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                pass
            elif token in [330]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2271
                self.match(fugue_sqlParser.STRING)
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

    class TablePropertyValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def booleanValue(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanValueContext, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tablePropertyValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTablePropertyValue'):
                return visitor.visitTablePropertyValue(self)
            else:
                return visitor.visitChildren(self)

    def tablePropertyValue(self):
        localctx = fugue_sqlParser.TablePropertyValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 212, self.RULE_tablePropertyValue)
        try:
            self.state = 2278
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [334]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2274
                self.match(fugue_sqlParser.INTEGER_VALUE)
                pass
            elif token in [336]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2275
                self.match(fugue_sqlParser.DECIMAL_VALUE)
                pass
            elif token in [135, 287]:
                self.enterOuterAlt(localctx, 3)
                self.state = 2276
                self.booleanValue()
                pass
            elif token in [330]:
                self.enterOuterAlt(localctx, 4)
                self.state = 2277
                self.match(fugue_sqlParser.STRING)
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
                return self.getTypedRuleContexts(fugue_sqlParser.ConstantContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_constantList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitConstantList'):
                return visitor.visitConstantList(self)
            else:
                return visitor.visitChildren(self)

    def constantList(self):
        localctx = fugue_sqlParser.ConstantListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 214, self.RULE_constantList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2280
            self.match(fugue_sqlParser.T__4)
            self.state = 2281
            self.constant()
            self.state = 2286
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2282
                self.match(fugue_sqlParser.T__1)
                self.state = 2283
                self.constant()
                self.state = 2288
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2289
            self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.ConstantListContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ConstantListContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_nestedConstantList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNestedConstantList'):
                return visitor.visitNestedConstantList(self)
            else:
                return visitor.visitChildren(self)

    def nestedConstantList(self):
        localctx = fugue_sqlParser.NestedConstantListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 216, self.RULE_nestedConstantList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2291
            self.match(fugue_sqlParser.T__4)
            self.state = 2292
            self.constantList()
            self.state = 2297
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2293
                self.match(fugue_sqlParser.T__1)
                self.state = 2294
                self.constantList()
                self.state = 2299
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2300
            self.match(fugue_sqlParser.T__5)
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
            return self.getToken(fugue_sqlParser.STORED, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def fileFormat(self):
            return self.getTypedRuleContext(fugue_sqlParser.FileFormatContext, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def storageHandler(self):
            return self.getTypedRuleContext(fugue_sqlParser.StorageHandlerContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_createFileFormat

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitCreateFileFormat'):
                return visitor.visitCreateFileFormat(self)
            else:
                return visitor.visitChildren(self)

    def createFileFormat(self):
        localctx = fugue_sqlParser.CreateFileFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 218, self.RULE_createFileFormat)
        try:
            self.state = 2308
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 286, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2302
                self.match(fugue_sqlParser.STORED)
                self.state = 2303
                self.match(fugue_sqlParser.AS)
                self.state = 2304
                self.fileFormat()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2305
                self.match(fugue_sqlParser.STORED)
                self.state = 2306
                self.match(fugue_sqlParser.BY)
                self.state = 2307
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
            return fugue_sqlParser.RULE_fileFormat

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class TableFileFormatContext(FileFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.inFmt = None
            self.outFmt = None
            self.copyFrom(ctx)

        def INPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.STRING)
            else:
                return self.getToken(fugue_sqlParser.STRING, i)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitGenericFileFormat'):
                return visitor.visitGenericFileFormat(self)
            else:
                return visitor.visitChildren(self)

    def fileFormat(self):
        localctx = fugue_sqlParser.FileFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 220, self.RULE_fileFormat)
        try:
            self.state = 2315
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 287, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.TableFileFormatContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2310
                self.match(fugue_sqlParser.INPUTFORMAT)
                self.state = 2311
                localctx.inFmt = self.match(fugue_sqlParser.STRING)
                self.state = 2312
                self.match(fugue_sqlParser.OUTPUTFORMAT)
                self.state = 2313
                localctx.outFmt = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.GenericFileFormatContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2314
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
            return self.getToken(fugue_sqlParser.STRING, 0)

        def WITH(self):
            return self.getToken(fugue_sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_storageHandler

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStorageHandler'):
                return visitor.visitStorageHandler(self)
            else:
                return visitor.visitChildren(self)

    def storageHandler(self):
        localctx = fugue_sqlParser.StorageHandlerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 222, self.RULE_storageHandler)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2317
            self.match(fugue_sqlParser.STRING)
            self.state = 2321
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 288, self._ctx)
            if la_ == 1:
                self.state = 2318
                self.match(fugue_sqlParser.WITH)
                self.state = 2319
                self.match(fugue_sqlParser.SERDEPROPERTIES)
                self.state = 2320
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_resource

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitResource'):
                return visitor.visitResource(self)
            else:
                return visitor.visitChildren(self)

    def resource(self):
        localctx = fugue_sqlParser.ResourceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 224, self.RULE_resource)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2323
            self.identifier()
            self.state = 2324
            self.match(fugue_sqlParser.STRING)
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
            return fugue_sqlParser.RULE_dmlStatementNoWith

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class DeleteFromTableContext(DmlStatementNoWithContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DELETE(self):
            return self.getToken(fugue_sqlParser.DELETE, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.InsertIntoContext, 0)

        def queryTerm(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryTermContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryOrganizationContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

        def multiInsertQueryBody(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultiInsertQueryBodyContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultiInsertQueryBodyContext, i)

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
            return self.getToken(fugue_sqlParser.UPDATE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def setClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.SetClauseContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

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
            return self.getToken(fugue_sqlParser.MERGE, 0)

        def INTO(self):
            return self.getToken(fugue_sqlParser.INTO, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def multipartIdentifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

        def tableAlias(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TableAliasContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, i)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def matchedClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.MatchedClauseContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MatchedClauseContext, i)

        def notMatchedClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.NotMatchedClauseContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NotMatchedClauseContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMergeIntoTable'):
                return visitor.visitMergeIntoTable(self)
            else:
                return visitor.visitChildren(self)

    def dmlStatementNoWith(self):
        localctx = fugue_sqlParser.DmlStatementNoWithContext(self, self._ctx, self.state)
        self.enterRule(localctx, 226, self.RULE_dmlStatementNoWith)
        self._la = 0
        try:
            self.state = 2377
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [165]:
                localctx = fugue_sqlParser.SingleInsertQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2326
                self.insertInto()
                self.state = 2327
                self.queryTerm(0)
                self.state = 2328
                self.queryOrganization()
                pass
            elif token in [146]:
                localctx = fugue_sqlParser.MultiInsertQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2330
                self.fromClause()
                self.state = 2332
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 2331
                    self.multiInsertQueryBody()
                    self.state = 2334
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 165:
                        break
                pass
            elif token in [113]:
                localctx = fugue_sqlParser.DeleteFromTableContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2336
                self.match(fugue_sqlParser.DELETE)
                self.state = 2337
                self.match(fugue_sqlParser.FROM)
                self.state = 2338
                self.multipartIdentifier()
                self.state = 2339
                self.tableAlias()
                self.state = 2341
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 306:
                    self.state = 2340
                    self.whereClause()
                pass
            elif token in [298]:
                localctx = fugue_sqlParser.UpdateTableContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2343
                self.match(fugue_sqlParser.UPDATE)
                self.state = 2344
                self.multipartIdentifier()
                self.state = 2345
                self.tableAlias()
                self.state = 2346
                self.setClause()
                self.state = 2348
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 306:
                    self.state = 2347
                    self.whereClause()
                pass
            elif token in [191]:
                localctx = fugue_sqlParser.MergeIntoTableContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 2350
                self.match(fugue_sqlParser.MERGE)
                self.state = 2351
                self.match(fugue_sqlParser.INTO)
                self.state = 2352
                localctx.target = self.multipartIdentifier()
                self.state = 2353
                localctx.targetAlias = self.tableAlias()
                self.state = 2354
                self.match(fugue_sqlParser.USING)
                self.state = 2360
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                    self.state = 2355
                    localctx.source = self.multipartIdentifier()
                    pass
                elif token in [5]:
                    self.state = 2356
                    self.match(fugue_sqlParser.T__4)
                    self.state = 2357
                    localctx.sourceQuery = self.query()
                    self.state = 2358
                    self.match(fugue_sqlParser.T__5)
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 2362
                localctx.sourceAlias = self.tableAlias()
                self.state = 2363
                self.match(fugue_sqlParser.ON)
                self.state = 2364
                localctx.mergeCondition = self.booleanExpression(0)
                self.state = 2368
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 293, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2365
                        self.matchedClause()
                    self.state = 2370
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 293, self._ctx)
                self.state = 2374
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 305:
                    self.state = 2371
                    self.notMatchedClause()
                    self.state = 2376
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
            return self.getToken(fugue_sqlParser.ORDER, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def CLUSTER(self):
            return self.getToken(fugue_sqlParser.CLUSTER, 0)

        def DISTRIBUTE(self):
            return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

        def SORT(self):
            return self.getToken(fugue_sqlParser.SORT, 0)

        def windowClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

        def LIMIT(self):
            return self.getToken(fugue_sqlParser.LIMIT, 0)

        def sortItem(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.SortItemContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.SortItemContext, i)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_queryOrganization

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQueryOrganization'):
                return visitor.visitQueryOrganization(self)
            else:
                return visitor.visitChildren(self)

    def queryOrganization(self):
        localctx = fugue_sqlParser.QueryOrganizationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 228, self.RULE_queryOrganization)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2389
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 297, self._ctx)
            if la_ == 1:
                self.state = 2379
                self.match(fugue_sqlParser.ORDER)
                self.state = 2380
                self.match(fugue_sqlParser.BY)
                self.state = 2381
                localctx._sortItem = self.sortItem()
                localctx.order.append(localctx._sortItem)
                self.state = 2386
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 296, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2382
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2383
                        localctx._sortItem = self.sortItem()
                        localctx.order.append(localctx._sortItem)
                    self.state = 2388
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 296, self._ctx)
            self.state = 2401
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 299, self._ctx)
            if la_ == 1:
                self.state = 2391
                self.match(fugue_sqlParser.CLUSTER)
                self.state = 2392
                self.match(fugue_sqlParser.BY)
                self.state = 2393
                localctx._expression = self.expression()
                localctx.clusterBy.append(localctx._expression)
                self.state = 2398
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 298, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2394
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2395
                        localctx._expression = self.expression()
                        localctx.clusterBy.append(localctx._expression)
                    self.state = 2400
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 298, self._ctx)
            self.state = 2413
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 301, self._ctx)
            if la_ == 1:
                self.state = 2403
                self.match(fugue_sqlParser.DISTRIBUTE)
                self.state = 2404
                self.match(fugue_sqlParser.BY)
                self.state = 2405
                localctx._expression = self.expression()
                localctx.distributeBy.append(localctx._expression)
                self.state = 2410
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 300, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2406
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2407
                        localctx._expression = self.expression()
                        localctx.distributeBy.append(localctx._expression)
                    self.state = 2412
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 300, self._ctx)
            self.state = 2425
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 303, self._ctx)
            if la_ == 1:
                self.state = 2415
                self.match(fugue_sqlParser.SORT)
                self.state = 2416
                self.match(fugue_sqlParser.BY)
                self.state = 2417
                localctx._sortItem = self.sortItem()
                localctx.sort.append(localctx._sortItem)
                self.state = 2422
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 302, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2418
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2419
                        localctx._sortItem = self.sortItem()
                        localctx.sort.append(localctx._sortItem)
                    self.state = 2424
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 302, self._ctx)
            self.state = 2428
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 304, self._ctx)
            if la_ == 1:
                self.state = 2427
                self.windowClause()
            self.state = 2435
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 306, self._ctx)
            if la_ == 1:
                self.state = 2430
                self.match(fugue_sqlParser.LIMIT)
                self.state = 2433
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 305, self._ctx)
                if la_ == 1:
                    self.state = 2431
                    self.match(fugue_sqlParser.ALL)
                    pass
                elif la_ == 2:
                    self.state = 2432
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
            return self.getTypedRuleContext(fugue_sqlParser.InsertIntoContext, 0)

        def fromStatementBody(self):
            return self.getTypedRuleContext(fugue_sqlParser.FromStatementBodyContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_multiInsertQueryBody

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultiInsertQueryBody'):
                return visitor.visitMultiInsertQueryBody(self)
            else:
                return visitor.visitChildren(self)

    def multiInsertQueryBody(self):
        localctx = fugue_sqlParser.MultiInsertQueryBodyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 230, self.RULE_multiInsertQueryBody)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2437
            self.insertInto()
            self.state = 2438
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
            return fugue_sqlParser.RULE_queryTerm

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class QueryTermDefaultContext(QueryTermContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def queryPrimary(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryPrimaryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQueryTermDefault'):
                return visitor.visitQueryTermDefault(self)
            else:
                return visitor.visitChildren(self)

    class FugueTermContext(QueryTermContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def fugueNestableTaskCollectionNoSelect(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskCollectionNoSelectContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFugueTerm'):
                return visitor.visitFugueTerm(self)
            else:
                return visitor.visitChildren(self)

    class SetOperationContext(QueryTermContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.theOperator = None
            self.right = None
            self.copyFrom(ctx)

        def queryTerm(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.QueryTermContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.QueryTermContext, i)

        def INTERSECT(self):
            return self.getToken(fugue_sqlParser.INTERSECT, 0)

        def UNION(self):
            return self.getToken(fugue_sqlParser.UNION, 0)

        def EXCEPT(self):
            return self.getToken(fugue_sqlParser.EXCEPT, 0)

        def SETMINUS(self):
            return self.getToken(fugue_sqlParser.SETMINUS, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.SetQuantifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetOperation'):
                return visitor.visitSetOperation(self)
            else:
                return visitor.visitChildren(self)

    def queryTerm(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = fugue_sqlParser.QueryTermContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 232
        self.enterRecursionRule(localctx, 232, self.RULE_queryTerm, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2443
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [146, 189, 233, 252, 273, 302]:
                localctx = fugue_sqlParser.QueryTermDefaultContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2441
                self.queryPrimary()
                pass
            elif token in [17, 18, 27, 33, 36, 48, 61, 99, 122, 182, 236, 285]:
                localctx = fugue_sqlParser.FugueTermContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 2442
                self.fugueNestableTaskCollectionNoSelect()
                pass
            else:
                raise NoViableAltException(self)
            self._ctx.stop = self._input.LT(-1)
            self.state = 2465
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 312, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 2463
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 311, self._ctx)
                    if la_ == 1:
                        localctx = fugue_sqlParser.SetOperationContext(self, fugue_sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 2445
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 3)')
                        self.state = 2446
                        localctx.theOperator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la == 127 or _la == 166 or _la == 259 or (_la == 293)):
                            localctx.theOperator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2448
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 60 or _la == 120:
                            self.state = 2447
                            self.setQuantifier()
                        self.state = 2450
                        localctx.right = self.queryTerm(4)
                        pass
                    elif la_ == 2:
                        localctx = fugue_sqlParser.SetOperationContext(self, fugue_sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 2451
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 2452
                        localctx.theOperator = self.match(fugue_sqlParser.INTERSECT)
                        self.state = 2454
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 60 or _la == 120:
                            self.state = 2453
                            self.setQuantifier()
                        self.state = 2456
                        localctx.right = self.queryTerm(3)
                        pass
                    elif la_ == 3:
                        localctx = fugue_sqlParser.SetOperationContext(self, fugue_sqlParser.QueryTermContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_queryTerm)
                        self.state = 2457
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 2458
                        localctx.theOperator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la == 127 or _la == 259 or _la == 293):
                            localctx.theOperator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 2460
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        if _la == 60 or _la == 120:
                            self.state = 2459
                            self.setQuantifier()
                        self.state = 2462
                        localctx.right = self.queryTerm(2)
                        pass
                self.state = 2467
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 312, self._ctx)
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
            return fugue_sqlParser.RULE_queryPrimary

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class QueryPrimaryDefaultContext(QueryPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def querySpecification(self):
            return self.getTypedRuleContext(fugue_sqlParser.QuerySpecificationContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.InlineTableContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.FromStatementContext, 0)

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
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTable'):
                return visitor.visitTable(self)
            else:
                return visitor.visitChildren(self)

    def queryPrimary(self):
        localctx = fugue_sqlParser.QueryPrimaryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 234, self.RULE_queryPrimary)
        try:
            self.state = 2473
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [189, 233, 252]:
                localctx = fugue_sqlParser.QueryPrimaryDefaultContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2468
                self.querySpecification()
                pass
            elif token in [146]:
                localctx = fugue_sqlParser.FromStmtContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2469
                self.fromStatement()
                pass
            elif token in [273]:
                localctx = fugue_sqlParser.TableContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2470
                self.match(fugue_sqlParser.TABLE)
                self.state = 2471
                self.multipartIdentifier()
                pass
            elif token in [302]:
                localctx = fugue_sqlParser.InlineTableDefault1Context(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2472
                self.inlineTable()
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
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def ASC(self):
            return self.getToken(fugue_sqlParser.ASC, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def LAST(self):
            return self.getToken(fugue_sqlParser.LAST, 0)

        def FIRST(self):
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_sortItem

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSortItem'):
                return visitor.visitSortItem(self)
            else:
                return visitor.visitChildren(self)

    def sortItem(self):
        localctx = fugue_sqlParser.SortItemContext(self, self._ctx, self.state)
        self.enterRule(localctx, 236, self.RULE_sortItem)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2475
            self.expression()
            self.state = 2477
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 314, self._ctx)
            if la_ == 1:
                self.state = 2476
                localctx.ordering = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 115):
                    localctx.ordering = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
            self.state = 2481
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 315, self._ctx)
            if la_ == 1:
                self.state = 2479
                self.match(fugue_sqlParser.THENULLS)
                self.state = 2480
                localctx.nullOrder = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 140 or _la == 173):
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
            return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

        def fromStatementBody(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FromStatementBodyContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FromStatementBodyContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fromStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromStatement'):
                return visitor.visitFromStatement(self)
            else:
                return visitor.visitChildren(self)

    def fromStatement(self):
        localctx = fugue_sqlParser.FromStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 238, self.RULE_fromStatement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2483
            self.fromClause()
            self.state = 2485
            self._errHandler.sync(self)
            _alt = 1
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2484
                    self.fromStatementBody()
                else:
                    raise NoViableAltException(self)
                self.state = 2487
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 316, self._ctx)
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
            return self.getTypedRuleContext(fugue_sqlParser.TransformClauseContext, 0)

        def queryOrganization(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryOrganizationContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

        def selectClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.SelectClauseContext, 0)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

        def aggregationClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.AggregationClauseContext, 0)

        def havingClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.HavingClauseContext, 0)

        def windowClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fromStatementBody

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromStatementBody'):
                return visitor.visitFromStatementBody(self)
            else:
                return visitor.visitChildren(self)

    def fromStatementBody(self):
        localctx = fugue_sqlParser.FromStatementBodyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 240, self.RULE_fromStatementBody)
        try:
            self.state = 2516
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 323, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2489
                self.transformClause()
                self.state = 2491
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 317, self._ctx)
                if la_ == 1:
                    self.state = 2490
                    self.whereClause()
                self.state = 2493
                self.queryOrganization()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2495
                self.selectClause()
                self.state = 2499
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 318, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2496
                        self.lateralView()
                    self.state = 2501
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 318, self._ctx)
                self.state = 2503
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 319, self._ctx)
                if la_ == 1:
                    self.state = 2502
                    self.whereClause()
                self.state = 2506
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 320, self._ctx)
                if la_ == 1:
                    self.state = 2505
                    self.aggregationClause()
                self.state = 2509
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 321, self._ctx)
                if la_ == 1:
                    self.state = 2508
                    self.havingClause()
                self.state = 2512
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 322, self._ctx)
                if la_ == 1:
                    self.state = 2511
                    self.windowClause()
                self.state = 2514
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
            return fugue_sqlParser.RULE_querySpecification

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class RegularQuerySpecificationContext(QuerySpecificationContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def selectClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.SelectClauseContext, 0)

        def optionalFromClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.OptionalFromClauseContext, 0)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

        def whereClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

        def aggregationClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.AggregationClauseContext, 0)

        def havingClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.HavingClauseContext, 0)

        def windowClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowClauseContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.TransformClauseContext, 0)

        def optionalFromClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.OptionalFromClauseContext, 0)

        def whereClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.WhereClauseContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformQuerySpecification'):
                return visitor.visitTransformQuerySpecification(self)
            else:
                return visitor.visitChildren(self)

    def querySpecification(self):
        localctx = fugue_sqlParser.QuerySpecificationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 242, self.RULE_querySpecification)
        try:
            self.state = 2543
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 330, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.TransformQuerySpecificationContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2518
                self.transformClause()
                self.state = 2519
                self.optionalFromClause()
                self.state = 2521
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 324, self._ctx)
                if la_ == 1:
                    self.state = 2520
                    self.whereClause()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.RegularQuerySpecificationContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2523
                self.selectClause()
                self.state = 2524
                self.optionalFromClause()
                self.state = 2528
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 325, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2525
                        self.lateralView()
                    self.state = 2530
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 325, self._ctx)
                self.state = 2532
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 326, self._ctx)
                if la_ == 1:
                    self.state = 2531
                    self.whereClause()
                self.state = 2535
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 327, self._ctx)
                if la_ == 1:
                    self.state = 2534
                    self.aggregationClause()
                self.state = 2538
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 328, self._ctx)
                if la_ == 1:
                    self.state = 2537
                    self.havingClause()
                self.state = 2541
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 329, self._ctx)
                if la_ == 1:
                    self.state = 2540
                    self.windowClause()
                pass
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OptionalFromClauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fromClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.FromClauseContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_optionalFromClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOptionalFromClause'):
                return visitor.visitOptionalFromClause(self)
            else:
                return visitor.visitChildren(self)

    def optionalFromClause(self):
        localctx = fugue_sqlParser.OptionalFromClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 244, self.RULE_optionalFromClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2546
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 331, self._ctx)
            if la_ == 1:
                self.state = 2545
                self.fromClause()
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
            return self.getToken(fugue_sqlParser.USING, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.STRING)
            else:
                return self.getToken(fugue_sqlParser.STRING, i)

        def SELECT(self):
            return self.getToken(fugue_sqlParser.SELECT, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

        def TRANSFORM(self):
            return self.getToken(fugue_sqlParser.TRANSFORM, 0)

        def MAP(self):
            return self.getToken(fugue_sqlParser.MAP, 0)

        def REDUCE(self):
            return self.getToken(fugue_sqlParser.REDUCE, 0)

        def RECORDWRITER(self):
            return self.getToken(fugue_sqlParser.RECORDWRITER, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def RECORDREADER(self):
            return self.getToken(fugue_sqlParser.RECORDREADER, 0)

        def rowFormat(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

        def identifierSeq(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

        def colTypeList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_transformClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformClause'):
                return visitor.visitTransformClause(self)
            else:
                return visitor.visitChildren(self)

    def transformClause(self):
        localctx = fugue_sqlParser.TransformClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 246, self.RULE_transformClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2558
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [252]:
                self.state = 2548
                self.match(fugue_sqlParser.SELECT)
                self.state = 2549
                localctx.kind = self.match(fugue_sqlParser.TRANSFORM)
                self.state = 2550
                self.match(fugue_sqlParser.T__4)
                self.state = 2551
                self.namedExpressionSeq()
                self.state = 2552
                self.match(fugue_sqlParser.T__5)
                pass
            elif token in [189]:
                self.state = 2554
                localctx.kind = self.match(fugue_sqlParser.MAP)
                self.state = 2555
                self.namedExpressionSeq()
                pass
            elif token in [233]:
                self.state = 2556
                localctx.kind = self.match(fugue_sqlParser.REDUCE)
                self.state = 2557
                self.namedExpressionSeq()
                pass
            else:
                raise NoViableAltException(self)
            self.state = 2561
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 248:
                self.state = 2560
                localctx.inRowFormat = self.rowFormat()
            self.state = 2565
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 231:
                self.state = 2563
                self.match(fugue_sqlParser.RECORDWRITER)
                self.state = 2564
                localctx.recordWriter = self.match(fugue_sqlParser.STRING)
            self.state = 2567
            self.match(fugue_sqlParser.USING)
            self.state = 2568
            localctx.script = self.match(fugue_sqlParser.STRING)
            self.state = 2581
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 337, self._ctx)
            if la_ == 1:
                self.state = 2569
                self.match(fugue_sqlParser.AS)
                self.state = 2579
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 336, self._ctx)
                if la_ == 1:
                    self.state = 2570
                    self.identifierSeq()
                    pass
                elif la_ == 2:
                    self.state = 2571
                    self.colTypeList()
                    pass
                elif la_ == 3:
                    self.state = 2572
                    self.match(fugue_sqlParser.T__4)
                    self.state = 2575
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 335, self._ctx)
                    if la_ == 1:
                        self.state = 2573
                        self.identifierSeq()
                        pass
                    elif la_ == 2:
                        self.state = 2574
                        self.colTypeList()
                        pass
                    self.state = 2577
                    self.match(fugue_sqlParser.T__5)
                    pass
            self.state = 2584
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 338, self._ctx)
            if la_ == 1:
                self.state = 2583
                localctx.outRowFormat = self.rowFormat()
            self.state = 2588
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 339, self._ctx)
            if la_ == 1:
                self.state = 2586
                self.match(fugue_sqlParser.RECORDREADER)
                self.state = 2587
                localctx.recordReader = self.match(fugue_sqlParser.STRING)
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
            return self.getToken(fugue_sqlParser.SELECT, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.SetQuantifierContext, 0)

        def hint(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.HintContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.HintContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_selectClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSelectClause'):
                return visitor.visitSelectClause(self)
            else:
                return visitor.visitChildren(self)

    def selectClause(self):
        localctx = fugue_sqlParser.SelectClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 248, self.RULE_selectClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2590
            self.match(fugue_sqlParser.SELECT)
            self.state = 2594
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 14:
                self.state = 2591
                localctx._hint = self.hint()
                localctx.hints.append(localctx._hint)
                self.state = 2596
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2598
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 341, self._ctx)
            if la_ == 1:
                self.state = 2597
                self.setQuantifier()
            self.state = 2600
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
            return self.getToken(fugue_sqlParser.SET, 0)

        def assignmentList(self):
            return self.getTypedRuleContext(fugue_sqlParser.AssignmentListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_setClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetClause'):
                return visitor.visitSetClause(self)
            else:
                return visitor.visitChildren(self)

    def setClause(self):
        localctx = fugue_sqlParser.SetClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 250, self.RULE_setClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2602
            self.match(fugue_sqlParser.SET)
            self.state = 2603
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
            return self.getToken(fugue_sqlParser.WHEN, 0)

        def MATCHED(self):
            return self.getToken(fugue_sqlParser.MATCHED, 0)

        def THEN(self):
            return self.getToken(fugue_sqlParser.THEN, 0)

        def matchedAction(self):
            return self.getTypedRuleContext(fugue_sqlParser.MatchedActionContext, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_matchedClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMatchedClause'):
                return visitor.visitMatchedClause(self)
            else:
                return visitor.visitChildren(self)

    def matchedClause(self):
        localctx = fugue_sqlParser.MatchedClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 252, self.RULE_matchedClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2605
            self.match(fugue_sqlParser.WHEN)
            self.state = 2606
            self.match(fugue_sqlParser.MATCHED)
            self.state = 2609
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 63:
                self.state = 2607
                self.match(fugue_sqlParser.AND)
                self.state = 2608
                localctx.matchedCond = self.booleanExpression(0)
            self.state = 2611
            self.match(fugue_sqlParser.THEN)
            self.state = 2612
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
            return self.getToken(fugue_sqlParser.WHEN, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def MATCHED(self):
            return self.getToken(fugue_sqlParser.MATCHED, 0)

        def THEN(self):
            return self.getToken(fugue_sqlParser.THEN, 0)

        def notMatchedAction(self):
            return self.getTypedRuleContext(fugue_sqlParser.NotMatchedActionContext, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_notMatchedClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNotMatchedClause'):
                return visitor.visitNotMatchedClause(self)
            else:
                return visitor.visitChildren(self)

    def notMatchedClause(self):
        localctx = fugue_sqlParser.NotMatchedClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 254, self.RULE_notMatchedClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2614
            self.match(fugue_sqlParser.WHEN)
            self.state = 2615
            self.match(fugue_sqlParser.NOT)
            self.state = 2616
            self.match(fugue_sqlParser.MATCHED)
            self.state = 2619
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 63:
                self.state = 2617
                self.match(fugue_sqlParser.AND)
                self.state = 2618
                localctx.notMatchedCond = self.booleanExpression(0)
            self.state = 2621
            self.match(fugue_sqlParser.THEN)
            self.state = 2622
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
            return self.getToken(fugue_sqlParser.DELETE, 0)

        def UPDATE(self):
            return self.getToken(fugue_sqlParser.UPDATE, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def assignmentList(self):
            return self.getTypedRuleContext(fugue_sqlParser.AssignmentListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_matchedAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMatchedAction'):
                return visitor.visitMatchedAction(self)
            else:
                return visitor.visitChildren(self)

    def matchedAction(self):
        localctx = fugue_sqlParser.MatchedActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 256, self.RULE_matchedAction)
        try:
            self.state = 2631
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 344, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2624
                self.match(fugue_sqlParser.DELETE)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2625
                self.match(fugue_sqlParser.UPDATE)
                self.state = 2626
                self.match(fugue_sqlParser.SET)
                self.state = 2627
                self.match(fugue_sqlParser.ASTERISK)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2628
                self.match(fugue_sqlParser.UPDATE)
                self.state = 2629
                self.match(fugue_sqlParser.SET)
                self.state = 2630
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
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def VALUES(self):
            return self.getToken(fugue_sqlParser.VALUES, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def multipartIdentifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_notMatchedAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNotMatchedAction'):
                return visitor.visitNotMatchedAction(self)
            else:
                return visitor.visitChildren(self)

    def notMatchedAction(self):
        localctx = fugue_sqlParser.NotMatchedActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 258, self.RULE_notMatchedAction)
        self._la = 0
        try:
            self.state = 2651
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 346, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2633
                self.match(fugue_sqlParser.INSERT)
                self.state = 2634
                self.match(fugue_sqlParser.ASTERISK)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2635
                self.match(fugue_sqlParser.INSERT)
                self.state = 2636
                self.match(fugue_sqlParser.T__4)
                self.state = 2637
                localctx.columns = self.multipartIdentifierList()
                self.state = 2638
                self.match(fugue_sqlParser.T__5)
                self.state = 2639
                self.match(fugue_sqlParser.VALUES)
                self.state = 2640
                self.match(fugue_sqlParser.T__4)
                self.state = 2641
                self.expression()
                self.state = 2646
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 2642
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2643
                    self.expression()
                    self.state = 2648
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2649
                self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.AssignmentContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.AssignmentContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_assignmentList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAssignmentList'):
                return visitor.visitAssignmentList(self)
            else:
                return visitor.visitChildren(self)

    def assignmentList(self):
        localctx = fugue_sqlParser.AssignmentListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 260, self.RULE_assignmentList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2653
            self.assignment()
            self.state = 2658
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2654
                self.match(fugue_sqlParser.T__1)
                self.state = 2655
                self.assignment()
                self.state = 2660
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

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_assignment

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAssignment'):
                return visitor.visitAssignment(self)
            else:
                return visitor.visitChildren(self)

    def assignment(self):
        localctx = fugue_sqlParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 262, self.RULE_assignment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2661
            localctx.key = self.multipartIdentifier()
            self.state = 2662
            self.match(fugue_sqlParser.EQUAL)
            self.state = 2663
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
            return self.getToken(fugue_sqlParser.WHERE, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_whereClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWhereClause'):
                return visitor.visitWhereClause(self)
            else:
                return visitor.visitChildren(self)

    def whereClause(self):
        localctx = fugue_sqlParser.WhereClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 264, self.RULE_whereClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2665
            self.match(fugue_sqlParser.WHERE)
            self.state = 2666
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
            return self.getToken(fugue_sqlParser.HAVING, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_havingClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHavingClause'):
                return visitor.visitHavingClause(self)
            else:
                return visitor.visitChildren(self)

    def havingClause(self):
        localctx = fugue_sqlParser.HavingClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 266, self.RULE_havingClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2668
            self.match(fugue_sqlParser.HAVING)
            self.state = 2669
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
                return self.getTypedRuleContexts(fugue_sqlParser.HintStatementContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.HintStatementContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_hint

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHint'):
                return visitor.visitHint(self)
            else:
                return visitor.visitChildren(self)

    def hint(self):
        localctx = fugue_sqlParser.HintContext(self, self._ctx, self.state)
        self.enterRule(localctx, 268, self.RULE_hint)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2671
            self.match(fugue_sqlParser.T__13)
            self.state = 2672
            localctx._hintStatement = self.hintStatement()
            localctx.hintStatements.append(localctx._hintStatement)
            self.state = 2679
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la & ~63 == 0 and 1 << _la & -288230376151711740 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & 18014398509481983 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0):
                self.state = 2674
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2:
                    self.state = 2673
                    self.match(fugue_sqlParser.T__1)
                self.state = 2676
                localctx._hintStatement = self.hintStatement()
                localctx.hintStatements.append(localctx._hintStatement)
                self.state = 2681
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2682
            self.match(fugue_sqlParser.T__14)
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def primaryExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PrimaryExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_hintStatement

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitHintStatement'):
                return visitor.visitHintStatement(self)
            else:
                return visitor.visitChildren(self)

    def hintStatement(self):
        localctx = fugue_sqlParser.HintStatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 270, self.RULE_hintStatement)
        self._la = 0
        try:
            self.state = 2697
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 351, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2684
                localctx.hintName = self.identifier()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2685
                localctx.hintName = self.identifier()
                self.state = 2686
                self.match(fugue_sqlParser.T__4)
                self.state = 2687
                localctx._primaryExpression = self.primaryExpression(0)
                localctx.parameters.append(localctx._primaryExpression)
                self.state = 2692
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 2688
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2689
                    localctx._primaryExpression = self.primaryExpression(0)
                    localctx.parameters.append(localctx._primaryExpression)
                    self.state = 2694
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2695
                self.match(fugue_sqlParser.T__5)
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
            return self.getToken(fugue_sqlParser.FROM, 0)

        def relation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.RelationContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.RelationContext, i)

        def lateralView(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.LateralViewContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.LateralViewContext, i)

        def pivotClause(self):
            return self.getTypedRuleContext(fugue_sqlParser.PivotClauseContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_fromClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFromClause'):
                return visitor.visitFromClause(self)
            else:
                return visitor.visitChildren(self)

    def fromClause(self):
        localctx = fugue_sqlParser.FromClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 272, self.RULE_fromClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2699
            self.match(fugue_sqlParser.FROM)
            self.state = 2700
            self.relation()
            self.state = 2705
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 352, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2701
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2702
                    self.relation()
                self.state = 2707
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 352, self._ctx)
            self.state = 2711
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 353, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2708
                    self.lateralView()
                self.state = 2713
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 353, self._ctx)
            self.state = 2715
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 354, self._ctx)
            if la_ == 1:
                self.state = 2714
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
            return self.getToken(fugue_sqlParser.GROUP, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def WITH(self):
            return self.getToken(fugue_sqlParser.WITH, 0)

        def SETS(self):
            return self.getToken(fugue_sqlParser.SETS, 0)

        def groupingSet(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.GroupingSetContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.GroupingSetContext, i)

        def ROLLUP(self):
            return self.getToken(fugue_sqlParser.ROLLUP, 0)

        def CUBE(self):
            return self.getToken(fugue_sqlParser.CUBE, 0)

        def GROUPING(self):
            return self.getToken(fugue_sqlParser.GROUPING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_aggregationClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAggregationClause'):
                return visitor.visitAggregationClause(self)
            else:
                return visitor.visitChildren(self)

    def aggregationClause(self):
        localctx = fugue_sqlParser.AggregationClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 274, self.RULE_aggregationClause)
        self._la = 0
        try:
            self.state = 2761
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 359, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2717
                self.match(fugue_sqlParser.GROUP)
                self.state = 2718
                self.match(fugue_sqlParser.BY)
                self.state = 2719
                localctx._expression = self.expression()
                localctx.groupingExpressions.append(localctx._expression)
                self.state = 2724
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 355, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2720
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2721
                        localctx._expression = self.expression()
                        localctx.groupingExpressions.append(localctx._expression)
                    self.state = 2726
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 355, self._ctx)
                self.state = 2744
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 357, self._ctx)
                if la_ == 1:
                    self.state = 2727
                    self.match(fugue_sqlParser.WITH)
                    self.state = 2728
                    localctx.kind = self.match(fugue_sqlParser.ROLLUP)
                elif la_ == 2:
                    self.state = 2729
                    self.match(fugue_sqlParser.WITH)
                    self.state = 2730
                    localctx.kind = self.match(fugue_sqlParser.CUBE)
                elif la_ == 3:
                    self.state = 2731
                    localctx.kind = self.match(fugue_sqlParser.GROUPING)
                    self.state = 2732
                    self.match(fugue_sqlParser.SETS)
                    self.state = 2733
                    self.match(fugue_sqlParser.T__4)
                    self.state = 2734
                    self.groupingSet()
                    self.state = 2739
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 2735
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2736
                        self.groupingSet()
                        self.state = 2741
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 2742
                    self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2746
                self.match(fugue_sqlParser.GROUP)
                self.state = 2747
                self.match(fugue_sqlParser.BY)
                self.state = 2748
                localctx.kind = self.match(fugue_sqlParser.GROUPING)
                self.state = 2749
                self.match(fugue_sqlParser.SETS)
                self.state = 2750
                self.match(fugue_sqlParser.T__4)
                self.state = 2751
                self.groupingSet()
                self.state = 2756
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 2752
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2753
                    self.groupingSet()
                    self.state = 2758
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2759
                self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_groupingSet

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitGroupingSet'):
                return visitor.visitGroupingSet(self)
            else:
                return visitor.visitChildren(self)

    def groupingSet(self):
        localctx = fugue_sqlParser.GroupingSetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 276, self.RULE_groupingSet)
        self._la = 0
        try:
            self.state = 2776
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 362, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2763
                self.match(fugue_sqlParser.T__4)
                self.state = 2772
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                    self.state = 2764
                    self.expression()
                    self.state = 2769
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 2765
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2766
                        self.expression()
                        self.state = 2771
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 2774
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2775
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
            return self.getToken(fugue_sqlParser.PIVOT, 0)

        def FOR(self):
            return self.getToken(fugue_sqlParser.FOR, 0)

        def pivotColumn(self):
            return self.getTypedRuleContext(fugue_sqlParser.PivotColumnContext, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def namedExpressionSeq(self):
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

        def pivotValue(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.PivotValueContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.PivotValueContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_pivotClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotClause'):
                return visitor.visitPivotClause(self)
            else:
                return visitor.visitChildren(self)

    def pivotClause(self):
        localctx = fugue_sqlParser.PivotClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 278, self.RULE_pivotClause)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2778
            self.match(fugue_sqlParser.PIVOT)
            self.state = 2779
            self.match(fugue_sqlParser.T__4)
            self.state = 2780
            localctx.aggregates = self.namedExpressionSeq()
            self.state = 2781
            self.match(fugue_sqlParser.FOR)
            self.state = 2782
            self.pivotColumn()
            self.state = 2783
            self.match(fugue_sqlParser.IN)
            self.state = 2784
            self.match(fugue_sqlParser.T__4)
            self.state = 2785
            localctx._pivotValue = self.pivotValue()
            localctx.pivotValues.append(localctx._pivotValue)
            self.state = 2790
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2786
                self.match(fugue_sqlParser.T__1)
                self.state = 2787
                localctx._pivotValue = self.pivotValue()
                localctx.pivotValues.append(localctx._pivotValue)
                self.state = 2792
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2793
            self.match(fugue_sqlParser.T__5)
            self.state = 2794
            self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_pivotColumn

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotColumn'):
                return visitor.visitPivotColumn(self)
            else:
                return visitor.visitChildren(self)

    def pivotColumn(self):
        localctx = fugue_sqlParser.PivotColumnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 280, self.RULE_pivotColumn)
        self._la = 0
        try:
            self.state = 2808
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2796
                localctx._identifier = self.identifier()
                localctx.identifiers.append(localctx._identifier)
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2797
                self.match(fugue_sqlParser.T__4)
                self.state = 2798
                localctx._identifier = self.identifier()
                localctx.identifiers.append(localctx._identifier)
                self.state = 2803
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 2799
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2800
                    localctx._identifier = self.identifier()
                    localctx.identifiers.append(localctx._identifier)
                    self.state = 2805
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 2806
                self.match(fugue_sqlParser.T__5)
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

    class PivotValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_pivotValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPivotValue'):
                return visitor.visitPivotValue(self)
            else:
                return visitor.visitChildren(self)

    def pivotValue(self):
        localctx = fugue_sqlParser.PivotValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 282, self.RULE_pivotValue)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2810
            self.expression()
            self.state = 2815
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la - 58 & ~63 == 0 and 1 << _la - 58 & -1 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -1 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -1 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1152921504606846975 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0):
                self.state = 2812
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 366, self._ctx)
                if la_ == 1:
                    self.state = 2811
                    self.match(fugue_sqlParser.AS)
                self.state = 2814
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
            return self.getToken(fugue_sqlParser.LATERAL, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def OUTER(self):
            return self.getToken(fugue_sqlParser.OUTER, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_lateralView

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLateralView'):
                return visitor.visitLateralView(self)
            else:
                return visitor.visitChildren(self)

    def lateralView(self):
        localctx = fugue_sqlParser.LateralViewContext(self, self._ctx, self.state)
        self.enterRule(localctx, 284, self.RULE_lateralView)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2817
            self.match(fugue_sqlParser.LATERAL)
            self.state = 2818
            self.match(fugue_sqlParser.VIEW)
            self.state = 2820
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 368, self._ctx)
            if la_ == 1:
                self.state = 2819
                self.match(fugue_sqlParser.OUTER)
            self.state = 2822
            self.qualifiedName()
            self.state = 2823
            self.match(fugue_sqlParser.T__4)
            self.state = 2832
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                self.state = 2824
                self.expression()
                self.state = 2829
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 2825
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2826
                    self.expression()
                    self.state = 2831
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
            self.state = 2834
            self.match(fugue_sqlParser.T__5)
            self.state = 2835
            localctx.tblName = self.identifier()
            self.state = 2847
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 373, self._ctx)
            if la_ == 1:
                self.state = 2837
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 371, self._ctx)
                if la_ == 1:
                    self.state = 2836
                    self.match(fugue_sqlParser.AS)
                self.state = 2839
                localctx._identifier = self.identifier()
                localctx.colName.append(localctx._identifier)
                self.state = 2844
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 372, self._ctx)
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 2840
                        self.match(fugue_sqlParser.T__1)
                        self.state = 2841
                        localctx._identifier = self.identifier()
                        localctx.colName.append(localctx._identifier)
                    self.state = 2846
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 372, self._ctx)
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
            return self.getToken(fugue_sqlParser.DISTINCT, 0)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_setQuantifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSetQuantifier'):
                return visitor.visitSetQuantifier(self)
            else:
                return visitor.visitChildren(self)

    def setQuantifier(self):
        localctx = fugue_sqlParser.SetQuantifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 286, self.RULE_setQuantifier)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2849
            _la = self._input.LA(1)
            if not (_la == 60 or _la == 120):
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
            return self.getTypedRuleContext(fugue_sqlParser.RelationPrimaryContext, 0)

        def joinRelation(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.JoinRelationContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.JoinRelationContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_relation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRelation'):
                return visitor.visitRelation(self)
            else:
                return visitor.visitChildren(self)

    def relation(self):
        localctx = fugue_sqlParser.RelationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 288, self.RULE_relation)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2851
            self.relationPrimary()
            self.state = 2855
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 374, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2852
                    self.joinRelation()
                self.state = 2857
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 374, self._ctx)
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

        def joinType(self):
            return self.getTypedRuleContext(fugue_sqlParser.JoinTypeContext, 0)

        def JOIN(self):
            return self.getToken(fugue_sqlParser.JOIN, 0)

        def relationPrimary(self):
            return self.getTypedRuleContext(fugue_sqlParser.RelationPrimaryContext, 0)

        def joinCriteria(self):
            return self.getTypedRuleContext(fugue_sqlParser.JoinCriteriaContext, 0)

        def NATURAL(self):
            return self.getToken(fugue_sqlParser.NATURAL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_joinRelation

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinRelation'):
                return visitor.visitJoinRelation(self)
            else:
                return visitor.visitChildren(self)

    def joinRelation(self):
        localctx = fugue_sqlParser.JoinRelationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 290, self.RULE_joinRelation)
        try:
            self.state = 2869
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [64, 100, 147, 162, 171, 177, 242, 253]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2858
                self.joinType()
                self.state = 2859
                self.match(fugue_sqlParser.JOIN)
                self.state = 2860
                localctx.right = self.relationPrimary()
                self.state = 2862
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 375, self._ctx)
                if la_ == 1:
                    self.state = 2861
                    self.joinCriteria()
                pass
            elif token in [197]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2864
                self.match(fugue_sqlParser.NATURAL)
                self.state = 2865
                self.joinType()
                self.state = 2866
                self.match(fugue_sqlParser.JOIN)
                self.state = 2867
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
            return self.getToken(fugue_sqlParser.INNER, 0)

        def CROSS(self):
            return self.getToken(fugue_sqlParser.CROSS, 0)

        def LEFT(self):
            return self.getToken(fugue_sqlParser.LEFT, 0)

        def OUTER(self):
            return self.getToken(fugue_sqlParser.OUTER, 0)

        def SEMI(self):
            return self.getToken(fugue_sqlParser.SEMI, 0)

        def RIGHT(self):
            return self.getToken(fugue_sqlParser.RIGHT, 0)

        def FULL(self):
            return self.getToken(fugue_sqlParser.FULL, 0)

        def ANTI(self):
            return self.getToken(fugue_sqlParser.ANTI, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_joinType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinType'):
                return visitor.visitJoinType(self)
            else:
                return visitor.visitChildren(self)

    def joinType(self):
        localctx = fugue_sqlParser.JoinTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 292, self.RULE_joinType)
        self._la = 0
        try:
            self.state = 2895
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 383, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 2872
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 162:
                    self.state = 2871
                    self.match(fugue_sqlParser.INNER)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 2874
                self.match(fugue_sqlParser.CROSS)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 2875
                self.match(fugue_sqlParser.LEFT)
                self.state = 2877
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 210:
                    self.state = 2876
                    self.match(fugue_sqlParser.OUTER)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 2880
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 177:
                    self.state = 2879
                    self.match(fugue_sqlParser.LEFT)
                self.state = 2882
                self.match(fugue_sqlParser.SEMI)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 2883
                self.match(fugue_sqlParser.RIGHT)
                self.state = 2885
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 210:
                    self.state = 2884
                    self.match(fugue_sqlParser.OUTER)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 2887
                self.match(fugue_sqlParser.FULL)
                self.state = 2889
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 210:
                    self.state = 2888
                    self.match(fugue_sqlParser.OUTER)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 2892
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 177:
                    self.state = 2891
                    self.match(fugue_sqlParser.LEFT)
                self.state = 2894
                self.match(fugue_sqlParser.ANTI)
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
            return self.getToken(fugue_sqlParser.ON, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_joinCriteria

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitJoinCriteria'):
                return visitor.visitJoinCriteria(self)
            else:
                return visitor.visitChildren(self)

    def joinCriteria(self):
        localctx = fugue_sqlParser.JoinCriteriaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 294, self.RULE_joinCriteria)
        try:
            self.state = 2901
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [203]:
                self.enterOuterAlt(localctx, 1)
                self.state = 2897
                self.match(fugue_sqlParser.ON)
                self.state = 2898
                self.booleanExpression(0)
                pass
            elif token in [301]:
                self.enterOuterAlt(localctx, 2)
                self.state = 2899
                self.match(fugue_sqlParser.USING)
                self.state = 2900
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
            return self.getToken(fugue_sqlParser.TABLESAMPLE, 0)

        def sampleMethod(self):
            return self.getTypedRuleContext(fugue_sqlParser.SampleMethodContext, 0)

        def SYSTEM(self):
            return self.getToken(fugue_sqlParser.SYSTEM, 0)

        def BERNOULLI(self):
            return self.getToken(fugue_sqlParser.BERNOULLI, 0)

        def RESERVOIR(self):
            return self.getToken(fugue_sqlParser.RESERVOIR, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_sample

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSample'):
                return visitor.visitSample(self)
            else:
                return visitor.visitChildren(self)

    def sample(self):
        localctx = fugue_sqlParser.SampleContext(self, self._ctx, self.state)
        self.enterRule(localctx, 296, self.RULE_sample)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2903
            self.match(fugue_sqlParser.TABLESAMPLE)
            self.state = 2905
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & 15762598695796736 != 0:
                self.state = 2904
                _la = self._input.LA(1)
                if not (_la & ~63 == 0 and 1 << _la & 15762598695796736 != 0):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
            self.state = 2907
            self.match(fugue_sqlParser.T__4)
            self.state = 2909
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                self.state = 2908
                self.sampleMethod()
            self.state = 2911
            self.match(fugue_sqlParser.T__5)
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
            return fugue_sqlParser.RULE_sampleMethod

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class SampleByRowsContext(SampleMethodContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

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
            return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

        def PERCENT(self):
            return self.getToken(fugue_sqlParser.PERCENT, 0)

        def INTEGER_VALUE(self):
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.OUT, 0)

        def OF(self):
            return self.getToken(fugue_sqlParser.OF, 0)

        def BUCKET(self):
            return self.getToken(fugue_sqlParser.BUCKET, 0)

        def INTEGER_VALUE(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.INTEGER_VALUE)
            else:
                return self.getToken(fugue_sqlParser.INTEGER_VALUE, i)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSampleByBytes'):
                return visitor.visitSampleByBytes(self)
            else:
                return visitor.visitChildren(self)

    def sampleMethod(self):
        localctx = fugue_sqlParser.SampleMethodContext(self, self._ctx, self.state)
        self.enterRule(localctx, 298, self.RULE_sampleMethod)
        self._la = 0
        try:
            self.state = 2937
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 390, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.SampleByPercentileContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2914
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 2913
                    localctx.negativeSign = self.match(fugue_sqlParser.MINUS)
                self.state = 2916
                localctx.percentage = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 334 or _la == 336):
                    localctx.percentage = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 2917
                _la = self._input.LA(1)
                if not (_la == 219 or _la == 323):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.SampleByRowsContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2918
                self.expression()
                self.state = 2919
                self.match(fugue_sqlParser.ROWS)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.SampleByBucketContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2921
                localctx.sampleType = self.match(fugue_sqlParser.BUCKET)
                self.state = 2922
                localctx.numerator = self.match(fugue_sqlParser.INTEGER_VALUE)
                self.state = 2923
                self.match(fugue_sqlParser.OUT)
                self.state = 2924
                self.match(fugue_sqlParser.OF)
                self.state = 2925
                localctx.denominator = self.match(fugue_sqlParser.INTEGER_VALUE)
                self.state = 2934
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 203:
                    self.state = 2926
                    self.match(fugue_sqlParser.ON)
                    self.state = 2932
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 388, self._ctx)
                    if la_ == 1:
                        self.state = 2927
                        self.identifier()
                        pass
                    elif la_ == 2:
                        self.state = 2928
                        self.qualifiedName()
                        self.state = 2929
                        self.match(fugue_sqlParser.T__4)
                        self.state = 2930
                        self.match(fugue_sqlParser.T__5)
                        pass
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.SampleByBytesContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 2936
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_identifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierList'):
                return visitor.visitIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def identifierList(self):
        localctx = fugue_sqlParser.IdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 300, self.RULE_identifierList)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2939
            self.match(fugue_sqlParser.T__4)
            self.state = 2940
            self.identifierSeq()
            self.state = 2941
            self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_identifierSeq

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierSeq'):
                return visitor.visitIdentifierSeq(self)
            else:
                return visitor.visitChildren(self)

    def identifierSeq(self):
        localctx = fugue_sqlParser.IdentifierSeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 302, self.RULE_identifierSeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2943
            localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
            localctx.ident.append(localctx._errorCapturingIdentifier)
            self.state = 2948
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 391, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 2944
                    self.match(fugue_sqlParser.T__1)
                    self.state = 2945
                    localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
                    localctx.ident.append(localctx._errorCapturingIdentifier)
                self.state = 2950
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 391, self._ctx)
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
                return self.getTypedRuleContexts(fugue_sqlParser.OrderedIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.OrderedIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_orderedIdentifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOrderedIdentifierList'):
                return visitor.visitOrderedIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def orderedIdentifierList(self):
        localctx = fugue_sqlParser.OrderedIdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 304, self.RULE_orderedIdentifierList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2951
            self.match(fugue_sqlParser.T__4)
            self.state = 2952
            self.orderedIdentifier()
            self.state = 2957
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2953
                self.match(fugue_sqlParser.T__1)
                self.state = 2954
                self.orderedIdentifier()
                self.state = 2959
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2960
            self.match(fugue_sqlParser.T__5)
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
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def ASC(self):
            return self.getToken(fugue_sqlParser.ASC, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_orderedIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitOrderedIdentifier'):
                return visitor.visitOrderedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def orderedIdentifier(self):
        localctx = fugue_sqlParser.OrderedIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 306, self.RULE_orderedIdentifier)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2962
            localctx.ident = self.errorCapturingIdentifier()
            self.state = 2964
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 69 or _la == 115:
                self.state = 2963
                localctx.ordering = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 69 or _la == 115):
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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierCommentContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierCommentContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_identifierCommentList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierCommentList'):
                return visitor.visitIdentifierCommentList(self)
            else:
                return visitor.visitChildren(self)

    def identifierCommentList(self):
        localctx = fugue_sqlParser.IdentifierCommentListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 308, self.RULE_identifierCommentList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2966
            self.match(fugue_sqlParser.T__4)
            self.state = 2967
            self.identifierComment()
            self.state = 2972
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 2968
                self.match(fugue_sqlParser.T__1)
                self.state = 2969
                self.identifierComment()
                self.state = 2974
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 2975
            self.match(fugue_sqlParser.T__5)
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_identifierComment

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifierComment'):
                return visitor.visitIdentifierComment(self)
            else:
                return visitor.visitChildren(self)

    def identifierComment(self):
        localctx = fugue_sqlParser.IdentifierCommentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 310, self.RULE_identifierComment)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 2977
            self.identifier()
            self.state = 2979
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 91:
                self.state = 2978
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
            return fugue_sqlParser.RULE_relationPrimary

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class TableValuedFunctionContext(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def functionTable(self):
            return self.getTypedRuleContext(fugue_sqlParser.FunctionTableContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableValuedFunction'):
                return visitor.visitTableValuedFunction(self)
            else:
                return visitor.visitChildren(self)

    class InlineTableDefault2Context(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def inlineTable(self):
            return self.getTypedRuleContext(fugue_sqlParser.InlineTableContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInlineTableDefault2'):
                return visitor.visitInlineTableDefault2(self)
            else:
                return visitor.visitChildren(self)

    class AliasedRelationContext(RelationPrimaryContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def relation(self):
            return self.getTypedRuleContext(fugue_sqlParser.RelationContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def sample(self):
            return self.getTypedRuleContext(fugue_sqlParser.SampleContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def sample(self):
            return self.getTypedRuleContext(fugue_sqlParser.SampleContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def fugueDataFrameMember(self):
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameMemberContext, 0)

        def sample(self):
            return self.getTypedRuleContext(fugue_sqlParser.SampleContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableName'):
                return visitor.visitTableName(self)
            else:
                return visitor.visitChildren(self)

    def relationPrimary(self):
        localctx = fugue_sqlParser.RelationPrimaryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 312, self.RULE_relationPrimary)
        try:
            self.state = 3008
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 400, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.TableNameContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 2981
                self.multipartIdentifier()
                self.state = 2983
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 396, self._ctx)
                if la_ == 1:
                    self.state = 2982
                    self.fugueDataFrameMember()
                self.state = 2986
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 397, self._ctx)
                if la_ == 1:
                    self.state = 2985
                    self.sample()
                self.state = 2988
                self.tableAlias()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.AliasedQueryContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 2990
                self.match(fugue_sqlParser.T__4)
                self.state = 2991
                self.query()
                self.state = 2992
                self.match(fugue_sqlParser.T__5)
                self.state = 2994
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 398, self._ctx)
                if la_ == 1:
                    self.state = 2993
                    self.sample()
                self.state = 2996
                self.tableAlias()
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.AliasedRelationContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 2998
                self.match(fugue_sqlParser.T__4)
                self.state = 2999
                self.relation()
                self.state = 3000
                self.match(fugue_sqlParser.T__5)
                self.state = 3002
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 399, self._ctx)
                if la_ == 1:
                    self.state = 3001
                    self.sample()
                self.state = 3004
                self.tableAlias()
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.InlineTableDefault2Context(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 3006
                self.inlineTable()
                pass
            elif la_ == 5:
                localctx = fugue_sqlParser.TableValuedFunctionContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 3007
                self.functionTable()
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
            return self.getToken(fugue_sqlParser.VALUES, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def tableAlias(self):
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_inlineTable

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInlineTable'):
                return visitor.visitInlineTable(self)
            else:
                return visitor.visitChildren(self)

    def inlineTable(self):
        localctx = fugue_sqlParser.InlineTableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 314, self.RULE_inlineTable)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3010
            self.match(fugue_sqlParser.VALUES)
            self.state = 3011
            self.expression()
            self.state = 3016
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 401, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3012
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3013
                    self.expression()
                self.state = 3018
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 401, self._ctx)
            self.state = 3019
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
            return self.getTypedRuleContext(fugue_sqlParser.TableAliasContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_functionTable

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionTable'):
                return visitor.visitFunctionTable(self)
            else:
                return visitor.visitChildren(self)

    def functionTable(self):
        localctx = fugue_sqlParser.FunctionTableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 316, self.RULE_functionTable)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3021
            localctx.funcName = self.errorCapturingIdentifier()
            self.state = 3022
            self.match(fugue_sqlParser.T__4)
            self.state = 3031
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                self.state = 3023
                self.expression()
                self.state = 3028
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 3024
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3025
                    self.expression()
                    self.state = 3030
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
            self.state = 3033
            self.match(fugue_sqlParser.T__5)
            self.state = 3034
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
            return self.getTypedRuleContext(fugue_sqlParser.StrictIdentifierContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tableAlias

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableAlias'):
                return visitor.visitTableAlias(self)
            else:
                return visitor.visitChildren(self)

    def tableAlias(self):
        localctx = fugue_sqlParser.TableAliasContext(self, self._ctx, self.state)
        self.enterRule(localctx, 318, self.RULE_tableAlias)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3043
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 406, self._ctx)
            if la_ == 1:
                self.state = 3037
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 404, self._ctx)
                if la_ == 1:
                    self.state = 3036
                    self.match(fugue_sqlParser.AS)
                self.state = 3039
                self.strictIdentifier()
                self.state = 3041
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 405, self._ctx)
                if la_ == 1:
                    self.state = 3040
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
            return fugue_sqlParser.RULE_rowFormat

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class RowFormatSerdeContext(RowFormatContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.name = None
            self.props = None
            self.copyFrom(ctx)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def FORMAT(self):
            return self.getToken(fugue_sqlParser.FORMAT, 0)

        def SERDE(self):
            return self.getToken(fugue_sqlParser.SERDE, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def WITH(self):
            return self.getToken(fugue_sqlParser.WITH, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

        def tablePropertyList(self):
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

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
            return self.getToken(fugue_sqlParser.ROW, 0)

        def FORMAT(self):
            return self.getToken(fugue_sqlParser.FORMAT, 0)

        def DELIMITED(self):
            return self.getToken(fugue_sqlParser.DELIMITED, 0)

        def FIELDS(self):
            return self.getToken(fugue_sqlParser.FIELDS, 0)

        def TERMINATED(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.TERMINATED)
            else:
                return self.getToken(fugue_sqlParser.TERMINATED, i)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def COLLECTION(self):
            return self.getToken(fugue_sqlParser.COLLECTION, 0)

        def ITEMS(self):
            return self.getToken(fugue_sqlParser.ITEMS, 0)

        def MAP(self):
            return self.getToken(fugue_sqlParser.MAP, 0)

        def KEYS(self):
            return self.getToken(fugue_sqlParser.KEYS, 0)

        def LINES(self):
            return self.getToken(fugue_sqlParser.LINES, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def DEFINED(self):
            return self.getToken(fugue_sqlParser.DEFINED, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def STRING(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.STRING)
            else:
                return self.getToken(fugue_sqlParser.STRING, i)

        def ESCAPED(self):
            return self.getToken(fugue_sqlParser.ESCAPED, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitRowFormatDelimited'):
                return visitor.visitRowFormatDelimited(self)
            else:
                return visitor.visitChildren(self)

    def rowFormat(self):
        localctx = fugue_sqlParser.RowFormatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 320, self.RULE_rowFormat)
        try:
            self.state = 3094
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 414, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.RowFormatSerdeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3045
                self.match(fugue_sqlParser.ROW)
                self.state = 3046
                self.match(fugue_sqlParser.FORMAT)
                self.state = 3047
                self.match(fugue_sqlParser.SERDE)
                self.state = 3048
                localctx.name = self.match(fugue_sqlParser.STRING)
                self.state = 3052
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 407, self._ctx)
                if la_ == 1:
                    self.state = 3049
                    self.match(fugue_sqlParser.WITH)
                    self.state = 3050
                    self.match(fugue_sqlParser.SERDEPROPERTIES)
                    self.state = 3051
                    localctx.props = self.tablePropertyList()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.RowFormatDelimitedContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3054
                self.match(fugue_sqlParser.ROW)
                self.state = 3055
                self.match(fugue_sqlParser.FORMAT)
                self.state = 3056
                self.match(fugue_sqlParser.DELIMITED)
                self.state = 3066
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 409, self._ctx)
                if la_ == 1:
                    self.state = 3057
                    self.match(fugue_sqlParser.FIELDS)
                    self.state = 3058
                    self.match(fugue_sqlParser.TERMINATED)
                    self.state = 3059
                    self.match(fugue_sqlParser.BY)
                    self.state = 3060
                    localctx.fieldsTerminatedBy = self.match(fugue_sqlParser.STRING)
                    self.state = 3064
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 408, self._ctx)
                    if la_ == 1:
                        self.state = 3061
                        self.match(fugue_sqlParser.ESCAPED)
                        self.state = 3062
                        self.match(fugue_sqlParser.BY)
                        self.state = 3063
                        localctx.escapedBy = self.match(fugue_sqlParser.STRING)
                self.state = 3073
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 410, self._ctx)
                if la_ == 1:
                    self.state = 3068
                    self.match(fugue_sqlParser.COLLECTION)
                    self.state = 3069
                    self.match(fugue_sqlParser.ITEMS)
                    self.state = 3070
                    self.match(fugue_sqlParser.TERMINATED)
                    self.state = 3071
                    self.match(fugue_sqlParser.BY)
                    self.state = 3072
                    localctx.collectionItemsTerminatedBy = self.match(fugue_sqlParser.STRING)
                self.state = 3080
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 411, self._ctx)
                if la_ == 1:
                    self.state = 3075
                    self.match(fugue_sqlParser.MAP)
                    self.state = 3076
                    self.match(fugue_sqlParser.KEYS)
                    self.state = 3077
                    self.match(fugue_sqlParser.TERMINATED)
                    self.state = 3078
                    self.match(fugue_sqlParser.BY)
                    self.state = 3079
                    localctx.keysTerminatedBy = self.match(fugue_sqlParser.STRING)
                self.state = 3086
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 412, self._ctx)
                if la_ == 1:
                    self.state = 3082
                    self.match(fugue_sqlParser.LINES)
                    self.state = 3083
                    self.match(fugue_sqlParser.TERMINATED)
                    self.state = 3084
                    self.match(fugue_sqlParser.BY)
                    self.state = 3085
                    localctx.linesSeparatedBy = self.match(fugue_sqlParser.STRING)
                self.state = 3092
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 413, self._ctx)
                if la_ == 1:
                    self.state = 3088
                    self.match(fugue_sqlParser.THENULL)
                    self.state = 3089
                    self.match(fugue_sqlParser.DEFINED)
                    self.state = 3090
                    self.match(fugue_sqlParser.AS)
                    self.state = 3091
                    localctx.nullDefinedAs = self.match(fugue_sqlParser.STRING)
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
                return self.getTypedRuleContexts(fugue_sqlParser.MultipartIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_multipartIdentifierList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultipartIdentifierList'):
                return visitor.visitMultipartIdentifierList(self)
            else:
                return visitor.visitChildren(self)

    def multipartIdentifierList(self):
        localctx = fugue_sqlParser.MultipartIdentifierListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 322, self.RULE_multipartIdentifierList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3096
            self.multipartIdentifier()
            self.state = 3101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 3097
                self.match(fugue_sqlParser.T__1)
                self.state = 3098
                self.multipartIdentifier()
                self.state = 3103
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
                return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_multipartIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultipartIdentifier'):
                return visitor.visitMultipartIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def multipartIdentifier(self):
        localctx = fugue_sqlParser.MultipartIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 324, self.RULE_multipartIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3104
            localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
            localctx.parts.append(localctx._errorCapturingIdentifier)
            self.state = 3109
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 416, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3105
                    self.match(fugue_sqlParser.T__6)
                    self.state = 3106
                    localctx._errorCapturingIdentifier = self.errorCapturingIdentifier()
                    localctx.parts.append(localctx._errorCapturingIdentifier)
                self.state = 3111
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 416, self._ctx)
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
                return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_tableIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTableIdentifier'):
                return visitor.visitTableIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def tableIdentifier(self):
        localctx = fugue_sqlParser.TableIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 326, self.RULE_tableIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3115
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 417, self._ctx)
            if la_ == 1:
                self.state = 3112
                localctx.db = self.errorCapturingIdentifier()
                self.state = 3113
                self.match(fugue_sqlParser.T__6)
            self.state = 3117
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
                return self.getTypedRuleContexts(fugue_sqlParser.ErrorCapturingIdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_functionIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionIdentifier'):
                return visitor.visitFunctionIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def functionIdentifier(self):
        localctx = fugue_sqlParser.FunctionIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 328, self.RULE_functionIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3122
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 418, self._ctx)
            if la_ == 1:
                self.state = 3119
                localctx.db = self.errorCapturingIdentifier()
                self.state = 3120
                self.match(fugue_sqlParser.T__6)
            self.state = 3124
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
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def identifierList(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_namedExpression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedExpression'):
                return visitor.visitNamedExpression(self)
            else:
                return visitor.visitChildren(self)

    def namedExpression(self):
        localctx = fugue_sqlParser.NamedExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 330, self.RULE_namedExpression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3126
            self.expression()
            self.state = 3134
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 421, self._ctx)
            if la_ == 1:
                self.state = 3128
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 419, self._ctx)
                if la_ == 1:
                    self.state = 3127
                    self.match(fugue_sqlParser.AS)
                self.state = 3132
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                    self.state = 3130
                    localctx.name = self.errorCapturingIdentifier()
                    pass
                elif token in [5]:
                    self.state = 3131
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

    class NamedExpressionSeqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def namedExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_namedExpressionSeq

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedExpressionSeq'):
                return visitor.visitNamedExpressionSeq(self)
            else:
                return visitor.visitChildren(self)

    def namedExpressionSeq(self):
        localctx = fugue_sqlParser.NamedExpressionSeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 332, self.RULE_namedExpressionSeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3136
            self.namedExpression()
            self.state = 3141
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 422, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3137
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3138
                    self.namedExpression()
                self.state = 3143
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 422, self._ctx)
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
                return self.getTypedRuleContexts(fugue_sqlParser.TransformContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TransformContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_transformList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformList'):
                return visitor.visitTransformList(self)
            else:
                return visitor.visitChildren(self)

    def transformList(self):
        localctx = fugue_sqlParser.TransformListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 334, self.RULE_transformList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3144
            self.match(fugue_sqlParser.T__4)
            self.state = 3145
            localctx._transform = self.transform()
            localctx.transforms.append(localctx._transform)
            self.state = 3150
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 3146
                self.match(fugue_sqlParser.T__1)
                self.state = 3147
                localctx._transform = self.transform()
                localctx.transforms.append(localctx._transform)
                self.state = 3152
                self._errHandler.sync(self)
                _la = self._input.LA(1)
            self.state = 3153
            self.match(fugue_sqlParser.T__5)
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
            return fugue_sqlParser.RULE_transform

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class IdentityTransformContext(TransformContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def qualifiedName(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def transformArgument(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.TransformArgumentContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.TransformArgumentContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitApplyTransform'):
                return visitor.visitApplyTransform(self)
            else:
                return visitor.visitChildren(self)

    def transform(self):
        localctx = fugue_sqlParser.TransformContext(self, self._ctx, self.state)
        self.enterRule(localctx, 336, self.RULE_transform)
        self._la = 0
        try:
            self.state = 3168
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 425, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.IdentityTransformContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3155
                self.qualifiedName()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.ApplyTransformContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3156
                localctx.transformName = self.identifier()
                self.state = 3157
                self.match(fugue_sqlParser.T__4)
                self.state = 3158
                localctx._transformArgument = self.transformArgument()
                localctx.argument.append(localctx._transformArgument)
                self.state = 3163
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 3159
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3160
                    localctx._transformArgument = self.transformArgument()
                    localctx.argument.append(localctx._transformArgument)
                    self.state = 3165
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 3166
                self.match(fugue_sqlParser.T__5)
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
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

        def constant(self):
            return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_transformArgument

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitTransformArgument'):
                return visitor.visitTransformArgument(self)
            else:
                return visitor.visitChildren(self)

    def transformArgument(self):
        localctx = fugue_sqlParser.TransformArgumentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 338, self.RULE_transformArgument)
        try:
            self.state = 3172
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 426, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3170
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3171
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
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_expression

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExpression'):
                return visitor.visitExpression(self)
            else:
                return visitor.visitChildren(self)

    def expression(self):
        localctx = fugue_sqlParser.ExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 340, self.RULE_expression)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3174
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
            return fugue_sqlParser.RULE_booleanExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class LogicalNotContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

        def predicate(self):
            return self.getTypedRuleContext(fugue_sqlParser.PredicateContext, 0)

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
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitExists'):
                return visitor.visitExists(self)
            else:
                return visitor.visitChildren(self)

    class LogicalBinaryContext(BooleanExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.theOperator = None
            self.right = None
            self.copyFrom(ctx)

        def booleanExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.BooleanExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, i)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitLogicalBinary'):
                return visitor.visitLogicalBinary(self)
            else:
                return visitor.visitChildren(self)

    def booleanExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = fugue_sqlParser.BooleanExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 342
        self.enterRecursionRule(localctx, 342, self.RULE_booleanExpression, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3188
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 428, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.LogicalNotContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3177
                self.match(fugue_sqlParser.NOT)
                self.state = 3178
                self.booleanExpression(5)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.ExistsContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3179
                self.match(fugue_sqlParser.EXISTS)
                self.state = 3180
                self.match(fugue_sqlParser.T__4)
                self.state = 3181
                self.query()
                self.state = 3182
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.PredicatedContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3184
                self.valueExpression(0)
                self.state = 3186
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 427, self._ctx)
                if la_ == 1:
                    self.state = 3185
                    self.predicate()
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 3198
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 430, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 3196
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 429, self._ctx)
                    if la_ == 1:
                        localctx = fugue_sqlParser.LogicalBinaryContext(self, fugue_sqlParser.BooleanExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_booleanExpression)
                        self.state = 3190
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 3191
                        localctx.theOperator = self.match(fugue_sqlParser.AND)
                        self.state = 3192
                        localctx.right = self.booleanExpression(3)
                        pass
                    elif la_ == 2:
                        localctx = fugue_sqlParser.LogicalBinaryContext(self, fugue_sqlParser.BooleanExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_booleanExpression)
                        self.state = 3193
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 3194
                        localctx.theOperator = self.match(fugue_sqlParser.OR)
                        self.state = 3195
                        localctx.right = self.booleanExpression(2)
                        pass
                self.state = 3200
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 430, self._ctx)
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
            return self.getToken(fugue_sqlParser.AND, 0)

        def BETWEEN(self):
            return self.getToken(fugue_sqlParser.BETWEEN, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def query(self):
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def RLIKE(self):
            return self.getToken(fugue_sqlParser.RLIKE, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def ANY(self):
            return self.getToken(fugue_sqlParser.ANY, 0)

        def SOME(self):
            return self.getToken(fugue_sqlParser.SOME, 0)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

        def ESCAPE(self):
            return self.getToken(fugue_sqlParser.ESCAPE, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def IS(self):
            return self.getToken(fugue_sqlParser.IS, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def TRUE(self):
            return self.getToken(fugue_sqlParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(fugue_sqlParser.FALSE, 0)

        def UNKNOWN(self):
            return self.getToken(fugue_sqlParser.UNKNOWN, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def DISTINCT(self):
            return self.getToken(fugue_sqlParser.DISTINCT, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_predicate

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPredicate'):
                return visitor.visitPredicate(self)
            else:
                return visitor.visitChildren(self)

    def predicate(self):
        localctx = fugue_sqlParser.PredicateContext(self, self._ctx, self.state)
        self.enterRule(localctx, 344, self.RULE_predicate)
        self._la = 0
        try:
            self.state = 3283
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 444, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3202
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3201
                    self.match(fugue_sqlParser.NOT)
                self.state = 3204
                localctx.kind = self.match(fugue_sqlParser.BETWEEN)
                self.state = 3205
                localctx.lower = self.valueExpression(0)
                self.state = 3206
                self.match(fugue_sqlParser.AND)
                self.state = 3207
                localctx.upper = self.valueExpression(0)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3210
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3209
                    self.match(fugue_sqlParser.NOT)
                self.state = 3212
                localctx.kind = self.match(fugue_sqlParser.IN)
                self.state = 3213
                self.match(fugue_sqlParser.T__4)
                self.state = 3214
                self.expression()
                self.state = 3219
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la == 2:
                    self.state = 3215
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3216
                    self.expression()
                    self.state = 3221
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                self.state = 3222
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 3225
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3224
                    self.match(fugue_sqlParser.NOT)
                self.state = 3227
                localctx.kind = self.match(fugue_sqlParser.IN)
                self.state = 3228
                self.match(fugue_sqlParser.T__4)
                self.state = 3229
                self.query()
                self.state = 3230
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 3233
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3232
                    self.match(fugue_sqlParser.NOT)
                self.state = 3235
                localctx.kind = self.match(fugue_sqlParser.RLIKE)
                self.state = 3236
                localctx.pattern = self.valueExpression(0)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 3238
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3237
                    self.match(fugue_sqlParser.NOT)
                self.state = 3240
                localctx.kind = self.match(fugue_sqlParser.LIKE)
                self.state = 3241
                localctx.quantifier = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 60 or _la == 65 or _la == 263):
                    localctx.quantifier = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 3255
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 438, self._ctx)
                if la_ == 1:
                    self.state = 3242
                    self.match(fugue_sqlParser.T__4)
                    self.state = 3243
                    self.match(fugue_sqlParser.T__5)
                    pass
                elif la_ == 2:
                    self.state = 3244
                    self.match(fugue_sqlParser.T__4)
                    self.state = 3245
                    self.expression()
                    self.state = 3250
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 3246
                        self.match(fugue_sqlParser.T__1)
                        self.state = 3247
                        self.expression()
                        self.state = 3252
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 3253
                    self.match(fugue_sqlParser.T__5)
                    pass
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 3258
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3257
                    self.match(fugue_sqlParser.NOT)
                self.state = 3260
                localctx.kind = self.match(fugue_sqlParser.LIKE)
                self.state = 3261
                localctx.pattern = self.valueExpression(0)
                self.state = 3264
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 440, self._ctx)
                if la_ == 1:
                    self.state = 3262
                    self.match(fugue_sqlParser.ESCAPE)
                    self.state = 3263
                    localctx.escapeChar = self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 3266
                self.match(fugue_sqlParser.IS)
                self.state = 3268
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3267
                    self.match(fugue_sqlParser.NOT)
                self.state = 3270
                localctx.kind = self.match(fugue_sqlParser.THENULL)
                pass
            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 3271
                self.match(fugue_sqlParser.IS)
                self.state = 3273
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3272
                    self.match(fugue_sqlParser.NOT)
                self.state = 3275
                localctx.kind = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 135 or _la == 287 or _la == 295):
                    localctx.kind = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 3276
                self.match(fugue_sqlParser.IS)
                self.state = 3278
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 199:
                    self.state = 3277
                    self.match(fugue_sqlParser.NOT)
                self.state = 3280
                localctx.kind = self.match(fugue_sqlParser.DISTINCT)
                self.state = 3281
                self.match(fugue_sqlParser.FROM)
                self.state = 3282
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
            return fugue_sqlParser.RULE_valueExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ValueExpressionDefaultContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def primaryExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.ComparisonOperatorContext, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComparison'):
                return visitor.visitComparison(self)
            else:
                return visitor.visitChildren(self)

    class ArithmeticBinaryContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.left = None
            self.theOperator = None
            self.right = None
            self.copyFrom(ctx)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def SLASH(self):
            return self.getToken(fugue_sqlParser.SLASH, 0)

        def PERCENT(self):
            return self.getToken(fugue_sqlParser.PERCENT, 0)

        def DIV(self):
            return self.getToken(fugue_sqlParser.DIV, 0)

        def PLUS(self):
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def CONCAT_PIPE(self):
            return self.getToken(fugue_sqlParser.CONCAT_PIPE, 0)

        def AMPERSAND(self):
            return self.getToken(fugue_sqlParser.AMPERSAND, 0)

        def HAT(self):
            return self.getToken(fugue_sqlParser.HAT, 0)

        def PIPE(self):
            return self.getToken(fugue_sqlParser.PIPE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticBinary'):
                return visitor.visitArithmeticBinary(self)
            else:
                return visitor.visitChildren(self)

    class ArithmeticUnaryContext(ValueExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.theOperator = None
            self.copyFrom(ctx)

        def valueExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def PLUS(self):
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def TILDE(self):
            return self.getToken(fugue_sqlParser.TILDE, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticUnary'):
                return visitor.visitArithmeticUnary(self)
            else:
                return visitor.visitChildren(self)

    def valueExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = fugue_sqlParser.ValueExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 346
        self.enterRecursionRule(localctx, 346, self.RULE_valueExpression, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3289
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 445, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.ValueExpressionDefaultContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3286
                self.primaryExpression(0)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.ArithmeticUnaryContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3287
                localctx.theOperator = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la - 319 & ~63 == 0 and 1 << _la - 319 & 67 != 0):
                    localctx.theOperator = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 3288
                self.valueExpression(7)
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 3312
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 447, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 3310
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 446, self._ctx)
                    if la_ == 1:
                        localctx = fugue_sqlParser.ArithmeticBinaryContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3291
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 6)')
                        self.state = 3292
                        localctx.theOperator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la - 321 & ~63 == 0 and 1 << _la - 321 & 15 != 0):
                            localctx.theOperator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 3293
                        localctx.right = self.valueExpression(7)
                        pass
                    elif la_ == 2:
                        localctx = fugue_sqlParser.ArithmeticBinaryContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3294
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 5)')
                        self.state = 3295
                        localctx.theOperator = self._input.LT(1)
                        _la = self._input.LA(1)
                        if not (_la - 319 & ~63 == 0 and 1 << _la - 319 & 515 != 0):
                            localctx.theOperator = self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 3296
                        localctx.right = self.valueExpression(6)
                        pass
                    elif la_ == 3:
                        localctx = fugue_sqlParser.ArithmeticBinaryContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3297
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 4)')
                        self.state = 3298
                        localctx.theOperator = self.match(fugue_sqlParser.AMPERSAND)
                        self.state = 3299
                        localctx.right = self.valueExpression(5)
                        pass
                    elif la_ == 4:
                        localctx = fugue_sqlParser.ArithmeticBinaryContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3300
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 3)')
                        self.state = 3301
                        localctx.theOperator = self.match(fugue_sqlParser.HAT)
                        self.state = 3302
                        localctx.right = self.valueExpression(4)
                        pass
                    elif la_ == 5:
                        localctx = fugue_sqlParser.ArithmeticBinaryContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3303
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 2)')
                        self.state = 3304
                        localctx.theOperator = self.match(fugue_sqlParser.PIPE)
                        self.state = 3305
                        localctx.right = self.valueExpression(3)
                        pass
                    elif la_ == 6:
                        localctx = fugue_sqlParser.ComparisonContext(self, fugue_sqlParser.ValueExpressionContext(self, _parentctx, _parentState))
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_valueExpression)
                        self.state = 3306
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 1)')
                        self.state = 3307
                        self.comparisonOperator()
                        self.state = 3308
                        localctx.right = self.valueExpression(2)
                        pass
                self.state = 3314
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 447, self._ctx)
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
            return fugue_sqlParser.RULE_primaryExpression

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class StructContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self._namedExpression = None
            self.argument = list()
            self.copyFrom(ctx)

        def STRUCT(self):
            return self.getToken(fugue_sqlParser.STRUCT, 0)

        def namedExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

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
            return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.CASE, 0)

        def END(self):
            return self.getToken(fugue_sqlParser.END, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def whenClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.WhenClauseContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.WhenClauseContext, i)

        def ELSE(self):
            return self.getToken(fugue_sqlParser.ELSE, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

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
                return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

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
            return self.getToken(fugue_sqlParser.LAST, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def IGNORE(self):
            return self.getToken(fugue_sqlParser.IGNORE, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

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
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def qualifiedName(self):
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStar'):
                return visitor.visitStar(self)
            else:
                return visitor.visitChildren(self)

    class OverlayContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.iinput = None
            self.replace = None
            self.position = None
            self.length = None
            self.copyFrom(ctx)

        def OVERLAY(self):
            return self.getToken(fugue_sqlParser.OVERLAY, 0)

        def PLACING(self):
            return self.getToken(fugue_sqlParser.PLACING, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def FOR(self):
            return self.getToken(fugue_sqlParser.FOR, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.PrimaryExpressionContext, 0)

        def valueExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSubqueryExpression'):
                return visitor.visitSubqueryExpression(self)
            else:
                return visitor.visitChildren(self)

    class SubstringContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.istr = None
            self.pos = None
            self.ilen = None
            self.copyFrom(ctx)

        def SUBSTR(self):
            return self.getToken(fugue_sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(fugue_sqlParser.SUBSTRING, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def FOR(self):
            return self.getToken(fugue_sqlParser.FOR, 0)

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
            return self.getToken(fugue_sqlParser.CURRENT_DATE, 0)

        def CURRENT_TIMESTAMP(self):
            return self.getToken(fugue_sqlParser.CURRENT_TIMESTAMP, 0)

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
            return self.getToken(fugue_sqlParser.CAST, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def dataType(self):
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.ConstantContext, 0)

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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

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
            return self.getToken(fugue_sqlParser.EXTRACT, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def valueExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

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
            return self.getToken(fugue_sqlParser.TRIM, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

        def BOTH(self):
            return self.getToken(fugue_sqlParser.BOTH, 0)

        def LEADING(self):
            return self.getToken(fugue_sqlParser.LEADING, 0)

        def TRAILING(self):
            return self.getToken(fugue_sqlParser.TRAILING, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.FunctionNameContext, 0)

        def FILTER(self):
            return self.getToken(fugue_sqlParser.FILTER, 0)

        def WHERE(self):
            return self.getToken(fugue_sqlParser.WHERE, 0)

        def OVER(self):
            return self.getToken(fugue_sqlParser.OVER, 0)

        def windowSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowSpecContext, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def booleanExpression(self):
            return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

        def setQuantifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.SetQuantifierContext, 0)

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
            return self.getToken(fugue_sqlParser.CASE, 0)

        def END(self):
            return self.getToken(fugue_sqlParser.END, 0)

        def whenClause(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.WhenClauseContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.WhenClauseContext, i)

        def ELSE(self):
            return self.getToken(fugue_sqlParser.ELSE, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSearchedCase'):
                return visitor.visitSearchedCase(self)
            else:
                return visitor.visitChildren(self)

    class PositionContext(PrimaryExpressionContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.substr = None
            self.istr = None
            self.copyFrom(ctx)

        def POSITION(self):
            return self.getToken(fugue_sqlParser.POSITION, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def valueExpression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

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
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def IGNORE(self):
            return self.getToken(fugue_sqlParser.IGNORE, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFirst'):
                return visitor.visitFirst(self)
            else:
                return visitor.visitChildren(self)

    def primaryExpression(self, _p: int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = fugue_sqlParser.PrimaryExpressionContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 348
        self.enterRecursionRule(localctx, 348, self.RULE_primaryExpression, _p)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3499
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 467, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.CurrentDatetimeContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3316
                localctx.name = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 103 or _la == 105):
                    localctx.name = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.SearchedCaseContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3317
                self.match(fugue_sqlParser.CASE)
                self.state = 3319
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 3318
                    self.whenClause()
                    self.state = 3321
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 305:
                        break
                self.state = 3325
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 123:
                    self.state = 3323
                    self.match(fugue_sqlParser.ELSE)
                    self.state = 3324
                    localctx.elseExpression = self.expression()
                self.state = 3327
                self.match(fugue_sqlParser.END)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.SimpleCaseContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3329
                self.match(fugue_sqlParser.CASE)
                self.state = 3330
                localctx.value = self.expression()
                self.state = 3332
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 3331
                    self.whenClause()
                    self.state = 3334
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 305:
                        break
                self.state = 3338
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 123:
                    self.state = 3336
                    self.match(fugue_sqlParser.ELSE)
                    self.state = 3337
                    localctx.elseExpression = self.expression()
                self.state = 3340
                self.match(fugue_sqlParser.END)
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.CastContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3342
                self.match(fugue_sqlParser.CAST)
                self.state = 3343
                self.match(fugue_sqlParser.T__4)
                self.state = 3344
                self.expression()
                self.state = 3345
                self.match(fugue_sqlParser.AS)
                self.state = 3346
                self.dataType()
                self.state = 3347
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 5:
                localctx = fugue_sqlParser.StructContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3349
                self.match(fugue_sqlParser.STRUCT)
                self.state = 3350
                self.match(fugue_sqlParser.T__4)
                self.state = 3359
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                    self.state = 3351
                    localctx._namedExpression = self.namedExpression()
                    localctx.argument.append(localctx._namedExpression)
                    self.state = 3356
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 3352
                        self.match(fugue_sqlParser.T__1)
                        self.state = 3353
                        localctx._namedExpression = self.namedExpression()
                        localctx.argument.append(localctx._namedExpression)
                        self.state = 3358
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 3361
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 6:
                localctx = fugue_sqlParser.FirstContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3362
                self.match(fugue_sqlParser.FIRST)
                self.state = 3363
                self.match(fugue_sqlParser.T__4)
                self.state = 3364
                self.expression()
                self.state = 3367
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 157:
                    self.state = 3365
                    self.match(fugue_sqlParser.IGNORE)
                    self.state = 3366
                    self.match(fugue_sqlParser.THENULLS)
                self.state = 3369
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 7:
                localctx = fugue_sqlParser.LastContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3371
                self.match(fugue_sqlParser.LAST)
                self.state = 3372
                self.match(fugue_sqlParser.T__4)
                self.state = 3373
                self.expression()
                self.state = 3376
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 157:
                    self.state = 3374
                    self.match(fugue_sqlParser.IGNORE)
                    self.state = 3375
                    self.match(fugue_sqlParser.THENULLS)
                self.state = 3378
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 8:
                localctx = fugue_sqlParser.PositionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3380
                self.match(fugue_sqlParser.POSITION)
                self.state = 3381
                self.match(fugue_sqlParser.T__4)
                self.state = 3382
                localctx.substr = self.valueExpression(0)
                self.state = 3383
                self.match(fugue_sqlParser.IN)
                self.state = 3384
                localctx.istr = self.valueExpression(0)
                self.state = 3385
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 9:
                localctx = fugue_sqlParser.ConstantDefaultContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3387
                self.constant()
                pass
            elif la_ == 10:
                localctx = fugue_sqlParser.StarContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3388
                self.match(fugue_sqlParser.ASTERISK)
                pass
            elif la_ == 11:
                localctx = fugue_sqlParser.StarContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3389
                self.qualifiedName()
                self.state = 3390
                self.match(fugue_sqlParser.T__6)
                self.state = 3391
                self.match(fugue_sqlParser.ASTERISK)
                pass
            elif la_ == 12:
                localctx = fugue_sqlParser.RowConstructorContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3393
                self.match(fugue_sqlParser.T__4)
                self.state = 3394
                self.namedExpression()
                self.state = 3397
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 3395
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3396
                    self.namedExpression()
                    self.state = 3399
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 2:
                        break
                self.state = 3401
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 13:
                localctx = fugue_sqlParser.SubqueryExpressionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3403
                self.match(fugue_sqlParser.T__4)
                self.state = 3404
                self.query()
                self.state = 3405
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 14:
                localctx = fugue_sqlParser.FunctionCallContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3407
                self.functionName()
                self.state = 3408
                self.match(fugue_sqlParser.T__4)
                self.state = 3420
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la & ~63 == 0 and 1 << _la & -288230376151711712 != 0 or (_la - 64 & ~63 == 0 and 1 << _la - 64 & -1 != 0) or (_la - 128 & ~63 == 0 and 1 << _la - 128 & -1 != 0) or (_la - 192 & ~63 == 0 and 1 << _la - 192 & -1 != 0) or (_la - 256 & ~63 == 0 and 1 << _la - 256 & -9205357638345293825 != 0) or (_la - 320 & ~63 == 0 and 1 << _la - 320 & 2096179 != 0):
                    self.state = 3410
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 457, self._ctx)
                    if la_ == 1:
                        self.state = 3409
                        self.setQuantifier()
                    self.state = 3412
                    localctx._expression = self.expression()
                    localctx.argument.append(localctx._expression)
                    self.state = 3417
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 3413
                        self.match(fugue_sqlParser.T__1)
                        self.state = 3414
                        localctx._expression = self.expression()
                        localctx.argument.append(localctx._expression)
                        self.state = 3419
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                self.state = 3422
                self.match(fugue_sqlParser.T__5)
                self.state = 3429
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 460, self._ctx)
                if la_ == 1:
                    self.state = 3423
                    self.match(fugue_sqlParser.FILTER)
                    self.state = 3424
                    self.match(fugue_sqlParser.T__4)
                    self.state = 3425
                    self.match(fugue_sqlParser.WHERE)
                    self.state = 3426
                    localctx.where = self.booleanExpression(0)
                    self.state = 3427
                    self.match(fugue_sqlParser.T__5)
                self.state = 3433
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 461, self._ctx)
                if la_ == 1:
                    self.state = 3431
                    self.match(fugue_sqlParser.OVER)
                    self.state = 3432
                    self.windowSpec()
                pass
            elif la_ == 15:
                localctx = fugue_sqlParser.LambdaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3435
                self.identifier()
                self.state = 3436
                self.match(fugue_sqlParser.T__15)
                self.state = 3437
                self.expression()
                pass
            elif la_ == 16:
                localctx = fugue_sqlParser.LambdaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3439
                self.match(fugue_sqlParser.T__4)
                self.state = 3440
                self.identifier()
                self.state = 3443
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 3441
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3442
                    self.identifier()
                    self.state = 3445
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not _la == 2:
                        break
                self.state = 3447
                self.match(fugue_sqlParser.T__5)
                self.state = 3448
                self.match(fugue_sqlParser.T__15)
                self.state = 3449
                self.expression()
                pass
            elif la_ == 17:
                localctx = fugue_sqlParser.ColumnReferenceContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3451
                self.identifier()
                pass
            elif la_ == 18:
                localctx = fugue_sqlParser.ParenthesizedExpressionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3452
                self.match(fugue_sqlParser.T__4)
                self.state = 3453
                self.expression()
                self.state = 3454
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 19:
                localctx = fugue_sqlParser.ExtractContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3456
                self.match(fugue_sqlParser.EXTRACT)
                self.state = 3457
                self.match(fugue_sqlParser.T__4)
                self.state = 3458
                localctx.field = self.identifier()
                self.state = 3459
                self.match(fugue_sqlParser.FROM)
                self.state = 3460
                localctx.source = self.valueExpression(0)
                self.state = 3461
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 20:
                localctx = fugue_sqlParser.SubstringContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3463
                _la = self._input.LA(1)
                if not (_la == 271 or _la == 272):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 3464
                self.match(fugue_sqlParser.T__4)
                self.state = 3465
                localctx.istr = self.valueExpression(0)
                self.state = 3466
                _la = self._input.LA(1)
                if not (_la == 2 or _la == 146):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 3467
                localctx.pos = self.valueExpression(0)
                self.state = 3470
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 2 or _la == 142:
                    self.state = 3468
                    _la = self._input.LA(1)
                    if not (_la == 2 or _la == 142):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 3469
                    localctx.ilen = self.valueExpression(0)
                self.state = 3472
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 21:
                localctx = fugue_sqlParser.TrimContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3474
                self.match(fugue_sqlParser.TRIM)
                self.state = 3475
                self.match(fugue_sqlParser.T__4)
                self.state = 3477
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 464, self._ctx)
                if la_ == 1:
                    self.state = 3476
                    localctx.trimOption = self._input.LT(1)
                    _la = self._input.LA(1)
                    if not (_la == 73 or _la == 176 or _la == 282):
                        localctx.trimOption = self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 3480
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 465, self._ctx)
                if la_ == 1:
                    self.state = 3479
                    localctx.trimStr = self.valueExpression(0)
                self.state = 3482
                self.match(fugue_sqlParser.FROM)
                self.state = 3483
                localctx.srcStr = self.valueExpression(0)
                self.state = 3484
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 22:
                localctx = fugue_sqlParser.OverlayContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 3486
                self.match(fugue_sqlParser.OVERLAY)
                self.state = 3487
                self.match(fugue_sqlParser.T__4)
                self.state = 3488
                localctx.iinput = self.valueExpression(0)
                self.state = 3489
                self.match(fugue_sqlParser.PLACING)
                self.state = 3490
                localctx.replace = self.valueExpression(0)
                self.state = 3491
                self.match(fugue_sqlParser.FROM)
                self.state = 3492
                localctx.position = self.valueExpression(0)
                self.state = 3495
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 142:
                    self.state = 3493
                    self.match(fugue_sqlParser.FOR)
                    self.state = 3494
                    localctx.length = self.valueExpression(0)
                self.state = 3497
                self.match(fugue_sqlParser.T__5)
                pass
            self._ctx.stop = self._input.LT(-1)
            self.state = 3511
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 469, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 3509
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input, 468, self._ctx)
                    if la_ == 1:
                        localctx = fugue_sqlParser.SubscriptContext(self, fugue_sqlParser.PrimaryExpressionContext(self, _parentctx, _parentState))
                        localctx.value = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_primaryExpression)
                        self.state = 3501
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 8)')
                        self.state = 3502
                        self.match(fugue_sqlParser.T__0)
                        self.state = 3503
                        localctx.index = self.valueExpression(0)
                        self.state = 3504
                        self.match(fugue_sqlParser.T__2)
                        pass
                    elif la_ == 2:
                        localctx = fugue_sqlParser.DereferenceContext(self, fugue_sqlParser.PrimaryExpressionContext(self, _parentctx, _parentState))
                        localctx.base = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_primaryExpression)
                        self.state = 3506
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, 'self.precpred(self._ctx, 6)')
                        self.state = 3507
                        self.match(fugue_sqlParser.T__6)
                        self.state = 3508
                        localctx.fieldName = self.identifier()
                        pass
                self.state = 3513
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 469, self._ctx)
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
            return fugue_sqlParser.RULE_constant

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class NullLiteralContext(ConstantContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

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
                return self.getTokens(fugue_sqlParser.STRING)
            else:
                return self.getToken(fugue_sqlParser.STRING, i)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.IntervalContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.NumberContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.BooleanValueContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBooleanLiteral'):
                return visitor.visitBooleanLiteral(self)
            else:
                return visitor.visitChildren(self)

    def constant(self):
        localctx = fugue_sqlParser.ConstantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 350, self.RULE_constant)
        try:
            self.state = 3526
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 471, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.NullLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3514
                self.match(fugue_sqlParser.THENULL)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.IntervalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3515
                self.interval()
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.TypeConstructorContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 3516
                self.identifier()
                self.state = 3517
                self.match(fugue_sqlParser.STRING)
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.NumericLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 3519
                self.number()
                pass
            elif la_ == 5:
                localctx = fugue_sqlParser.BooleanLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 3520
                self.booleanValue()
                pass
            elif la_ == 6:
                localctx = fugue_sqlParser.StringLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 3522
                self._errHandler.sync(self)
                _alt = 1
                while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 3521
                        self.match(fugue_sqlParser.STRING)
                    else:
                        raise NoViableAltException(self)
                    self.state = 3524
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input, 470, self._ctx)
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

        def comparisonEqualOperator(self):
            return self.getTypedRuleContext(fugue_sqlParser.ComparisonEqualOperatorContext, 0)

        def NEQ(self):
            return self.getToken(fugue_sqlParser.NEQ, 0)

        def NEQJ(self):
            return self.getToken(fugue_sqlParser.NEQJ, 0)

        def LT(self):
            return self.getToken(fugue_sqlParser.LT, 0)

        def LTE(self):
            return self.getToken(fugue_sqlParser.LTE, 0)

        def GT(self):
            return self.getToken(fugue_sqlParser.GT, 0)

        def GTE(self):
            return self.getToken(fugue_sqlParser.GTE, 0)

        def NSEQ(self):
            return self.getToken(fugue_sqlParser.NSEQ, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_comparisonOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComparisonOperator'):
                return visitor.visitComparisonOperator(self)
            else:
                return visitor.visitChildren(self)

    def comparisonOperator(self):
        localctx = fugue_sqlParser.ComparisonOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 352, self.RULE_comparisonOperator)
        try:
            self.state = 3536
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [310, 311]:
                self.enterOuterAlt(localctx, 1)
                self.state = 3528
                self.comparisonEqualOperator()
                pass
            elif token in [313]:
                self.enterOuterAlt(localctx, 2)
                self.state = 3529
                self.match(fugue_sqlParser.NEQ)
                pass
            elif token in [314]:
                self.enterOuterAlt(localctx, 3)
                self.state = 3530
                self.match(fugue_sqlParser.NEQJ)
                pass
            elif token in [315]:
                self.enterOuterAlt(localctx, 4)
                self.state = 3531
                self.match(fugue_sqlParser.LT)
                pass
            elif token in [316]:
                self.enterOuterAlt(localctx, 5)
                self.state = 3532
                self.match(fugue_sqlParser.LTE)
                pass
            elif token in [317]:
                self.enterOuterAlt(localctx, 6)
                self.state = 3533
                self.match(fugue_sqlParser.GT)
                pass
            elif token in [318]:
                self.enterOuterAlt(localctx, 7)
                self.state = 3534
                self.match(fugue_sqlParser.GTE)
                pass
            elif token in [312]:
                self.enterOuterAlt(localctx, 8)
                self.state = 3535
                self.match(fugue_sqlParser.NSEQ)
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

    class ComparisonEqualOperatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DOUBLEEQUAL(self):
            return self.getToken(fugue_sqlParser.DOUBLEEQUAL, 0)

        def EQUAL(self):
            return self.getToken(fugue_sqlParser.EQUAL, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_comparisonEqualOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComparisonEqualOperator'):
                return visitor.visitComparisonEqualOperator(self)
            else:
                return visitor.visitChildren(self)

    def comparisonEqualOperator(self):
        localctx = fugue_sqlParser.ComparisonEqualOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 354, self.RULE_comparisonEqualOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3538
            _la = self._input.LA(1)
            if not (_la == 310 or _la == 311):
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
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def ASTERISK(self):
            return self.getToken(fugue_sqlParser.ASTERISK, 0)

        def SLASH(self):
            return self.getToken(fugue_sqlParser.SLASH, 0)

        def PERCENT(self):
            return self.getToken(fugue_sqlParser.PERCENT, 0)

        def DIV(self):
            return self.getToken(fugue_sqlParser.DIV, 0)

        def TILDE(self):
            return self.getToken(fugue_sqlParser.TILDE, 0)

        def AMPERSAND(self):
            return self.getToken(fugue_sqlParser.AMPERSAND, 0)

        def PIPE(self):
            return self.getToken(fugue_sqlParser.PIPE, 0)

        def CONCAT_PIPE(self):
            return self.getToken(fugue_sqlParser.CONCAT_PIPE, 0)

        def HAT(self):
            return self.getToken(fugue_sqlParser.HAT, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_arithmeticOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitArithmeticOperator'):
                return visitor.visitArithmeticOperator(self)
            else:
                return visitor.visitChildren(self)

    def arithmeticOperator(self):
        localctx = fugue_sqlParser.ArithmeticOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 356, self.RULE_arithmeticOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3540
            _la = self._input.LA(1)
            if not (_la - 319 & ~63 == 0 and 1 << _la - 319 & 2047 != 0):
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
            return self.getToken(fugue_sqlParser.OR, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_predicateOperator

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPredicateOperator'):
                return visitor.visitPredicateOperator(self)
            else:
                return visitor.visitChildren(self)

    def predicateOperator(self):
        localctx = fugue_sqlParser.PredicateOperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 358, self.RULE_predicateOperator)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3542
            _la = self._input.LA(1)
            if not (_la == 63 or (_la - 159 & ~63 == 0 and 1 << _la - 159 & 282574488338433 != 0)):
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
            return self.getToken(fugue_sqlParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(fugue_sqlParser.FALSE, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_booleanValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitBooleanValue'):
                return visitor.visitBooleanValue(self)
            else:
                return visitor.visitChildren(self)

    def booleanValue(self):
        localctx = fugue_sqlParser.BooleanValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 360, self.RULE_booleanValue)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3544
            _la = self._input.LA(1)
            if not (_la == 135 or _la == 287):
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
            return self.getToken(fugue_sqlParser.INTERVAL, 0)

        def errorCapturingMultiUnitsInterval(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingMultiUnitsIntervalContext, 0)

        def errorCapturingUnitToUnitInterval(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingUnitToUnitIntervalContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_interval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitInterval'):
                return visitor.visitInterval(self)
            else:
                return visitor.visitChildren(self)

    def interval(self):
        localctx = fugue_sqlParser.IntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 362, self.RULE_interval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3546
            self.match(fugue_sqlParser.INTERVAL)
            self.state = 3549
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 473, self._ctx)
            if la_ == 1:
                self.state = 3547
                self.errorCapturingMultiUnitsInterval()
            elif la_ == 2:
                self.state = 3548
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
            return self.getTypedRuleContext(fugue_sqlParser.MultiUnitsIntervalContext, 0)

        def unitToUnitInterval(self):
            return self.getTypedRuleContext(fugue_sqlParser.UnitToUnitIntervalContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_errorCapturingMultiUnitsInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingMultiUnitsInterval'):
                return visitor.visitErrorCapturingMultiUnitsInterval(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingMultiUnitsInterval(self):
        localctx = fugue_sqlParser.ErrorCapturingMultiUnitsIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 364, self.RULE_errorCapturingMultiUnitsInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3551
            self.multiUnitsInterval()
            self.state = 3553
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 474, self._ctx)
            if la_ == 1:
                self.state = 3552
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
                return self.getTypedRuleContexts(fugue_sqlParser.IntervalValueContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IntervalValueContext, i)

        def intervalUnit(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IntervalUnitContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IntervalUnitContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_multiUnitsInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitMultiUnitsInterval'):
                return visitor.visitMultiUnitsInterval(self)
            else:
                return visitor.visitChildren(self)

    def multiUnitsInterval(self):
        localctx = fugue_sqlParser.MultiUnitsIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 366, self.RULE_multiUnitsInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3558
            self._errHandler.sync(self)
            _alt = 1
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3555
                    self.intervalValue()
                    self.state = 3556
                    self.intervalUnit()
                else:
                    raise NoViableAltException(self)
                self.state = 3560
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 475, self._ctx)
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
                return self.getTypedRuleContexts(fugue_sqlParser.UnitToUnitIntervalContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.UnitToUnitIntervalContext, i)

        def multiUnitsInterval(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultiUnitsIntervalContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_errorCapturingUnitToUnitInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingUnitToUnitInterval'):
                return visitor.visitErrorCapturingUnitToUnitInterval(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingUnitToUnitInterval(self):
        localctx = fugue_sqlParser.ErrorCapturingUnitToUnitIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 368, self.RULE_errorCapturingUnitToUnitInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3562
            localctx.body = self.unitToUnitInterval()
            self.state = 3565
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 476, self._ctx)
            if la_ == 1:
                self.state = 3563
                localctx.error1 = self.multiUnitsInterval()
            elif la_ == 2:
                self.state = 3564
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
            self.ifrom = None
            self.to = None

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def intervalValue(self):
            return self.getTypedRuleContext(fugue_sqlParser.IntervalValueContext, 0)

        def intervalUnit(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IntervalUnitContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IntervalUnitContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_unitToUnitInterval

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnitToUnitInterval'):
                return visitor.visitUnitToUnitInterval(self)
            else:
                return visitor.visitChildren(self)

    def unitToUnitInterval(self):
        localctx = fugue_sqlParser.UnitToUnitIntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 370, self.RULE_unitToUnitInterval)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3567
            localctx.value = self.intervalValue()
            self.state = 3568
            localctx.ifrom = self.intervalUnit()
            self.state = 3569
            self.match(fugue_sqlParser.TO)
            self.state = 3570
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
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def PLUS(self):
            return self.getToken(fugue_sqlParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def STRING(self):
            return self.getToken(fugue_sqlParser.STRING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_intervalValue

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntervalValue'):
                return visitor.visitIntervalValue(self)
            else:
                return visitor.visitChildren(self)

    def intervalValue(self):
        localctx = fugue_sqlParser.IntervalValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 372, self.RULE_intervalValue)
        self._la = 0
        try:
            self.state = 3577
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [319, 320, 334, 336]:
                self.enterOuterAlt(localctx, 1)
                self.state = 3573
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 319 or _la == 320:
                    self.state = 3572
                    _la = self._input.LA(1)
                    if not (_la == 319 or _la == 320):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                self.state = 3575
                _la = self._input.LA(1)
                if not (_la == 334 or _la == 336):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif token in [330]:
                self.enterOuterAlt(localctx, 2)
                self.state = 3576
                self.match(fugue_sqlParser.STRING)
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
            return self.getToken(fugue_sqlParser.DAY, 0)

        def HOUR(self):
            return self.getToken(fugue_sqlParser.HOUR, 0)

        def MINUTE(self):
            return self.getToken(fugue_sqlParser.MINUTE, 0)

        def MONTH(self):
            return self.getToken(fugue_sqlParser.MONTH, 0)

        def SECOND(self):
            return self.getToken(fugue_sqlParser.SECOND, 0)

        def YEAR(self):
            return self.getToken(fugue_sqlParser.YEAR, 0)

        def identifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_intervalUnit

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIntervalUnit'):
                return visitor.visitIntervalUnit(self)
            else:
                return visitor.visitChildren(self)

    def intervalUnit(self):
        localctx = fugue_sqlParser.IntervalUnitContext(self, self._ctx, self.state)
        self.enterRule(localctx, 374, self.RULE_intervalUnit)
        try:
            self.state = 3586
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 479, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3579
                self.match(fugue_sqlParser.DAY)
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3580
                self.match(fugue_sqlParser.HOUR)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 3581
                self.match(fugue_sqlParser.MINUTE)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 3582
                self.match(fugue_sqlParser.MONTH)
                pass
            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 3583
                self.match(fugue_sqlParser.SECOND)
                pass
            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 3584
                self.match(fugue_sqlParser.YEAR)
                pass
            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 3585
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
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def AFTER(self):
            return self.getToken(fugue_sqlParser.AFTER, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_colPosition

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColPosition'):
                return visitor.visitColPosition(self)
            else:
                return visitor.visitChildren(self)

    def colPosition(self):
        localctx = fugue_sqlParser.ColPositionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 376, self.RULE_colPosition)
        try:
            self.state = 3591
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [140]:
                self.enterOuterAlt(localctx, 1)
                self.state = 3588
                localctx.position = self.match(fugue_sqlParser.FIRST)
                pass
            elif token in [59]:
                self.enterOuterAlt(localctx, 2)
                self.state = 3589
                localctx.position = self.match(fugue_sqlParser.AFTER)
                self.state = 3590
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
            return fugue_sqlParser.RULE_dataType

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ComplexDataTypeContext(DataTypeContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.icomplex = None
            self.copyFrom(ctx)

        def LT(self):
            return self.getToken(fugue_sqlParser.LT, 0)

        def dataType(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.DataTypeContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, i)

        def GT(self):
            return self.getToken(fugue_sqlParser.GT, 0)

        def ARRAY(self):
            return self.getToken(fugue_sqlParser.ARRAY, 0)

        def MAP(self):
            return self.getToken(fugue_sqlParser.MAP, 0)

        def STRUCT(self):
            return self.getToken(fugue_sqlParser.STRUCT, 0)

        def NEQ(self):
            return self.getToken(fugue_sqlParser.NEQ, 0)

        def complexColTypeList(self):
            return self.getTypedRuleContext(fugue_sqlParser.ComplexColTypeListContext, 0)

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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def INTEGER_VALUE(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.INTEGER_VALUE)
            else:
                return self.getToken(fugue_sqlParser.INTEGER_VALUE, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitPrimitiveDataType'):
                return visitor.visitPrimitiveDataType(self)
            else:
                return visitor.visitChildren(self)

    def dataType(self):
        localctx = fugue_sqlParser.DataTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 378, self.RULE_dataType)
        self._la = 0
        try:
            self.state = 3627
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 485, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3593
                localctx.icomplex = self.match(fugue_sqlParser.ARRAY)
                self.state = 3594
                self.match(fugue_sqlParser.LT)
                self.state = 3595
                self.dataType()
                self.state = 3596
                self.match(fugue_sqlParser.GT)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3598
                localctx.icomplex = self.match(fugue_sqlParser.MAP)
                self.state = 3599
                self.match(fugue_sqlParser.LT)
                self.state = 3600
                self.dataType()
                self.state = 3601
                self.match(fugue_sqlParser.T__1)
                self.state = 3602
                self.dataType()
                self.state = 3603
                self.match(fugue_sqlParser.GT)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.ComplexDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 3605
                localctx.icomplex = self.match(fugue_sqlParser.STRUCT)
                self.state = 3612
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [315]:
                    self.state = 3606
                    self.match(fugue_sqlParser.LT)
                    self.state = 3608
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la - 58 & ~63 == 0 and 1 << _la - 58 & -1 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -1 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -1 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1152921504606846975 != 0) or (_la - 324 & ~63 == 0 and 1 << _la - 324 & 98305 != 0):
                        self.state = 3607
                        self.complexColTypeList()
                    self.state = 3610
                    self.match(fugue_sqlParser.GT)
                    pass
                elif token in [313]:
                    self.state = 3611
                    self.match(fugue_sqlParser.NEQ)
                    pass
                else:
                    raise NoViableAltException(self)
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.PrimitiveDataTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 3614
                self.identifier()
                self.state = 3625
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input, 484, self._ctx)
                if la_ == 1:
                    self.state = 3615
                    self.match(fugue_sqlParser.T__4)
                    self.state = 3616
                    self.match(fugue_sqlParser.INTEGER_VALUE)
                    self.state = 3621
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 3617
                        self.match(fugue_sqlParser.T__1)
                        self.state = 3618
                        self.match(fugue_sqlParser.INTEGER_VALUE)
                        self.state = 3623
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    self.state = 3624
                    self.match(fugue_sqlParser.T__5)
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
                return self.getTypedRuleContexts(fugue_sqlParser.QualifiedColTypeWithPositionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.QualifiedColTypeWithPositionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_qualifiedColTypeWithPositionList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedColTypeWithPositionList'):
                return visitor.visitQualifiedColTypeWithPositionList(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedColTypeWithPositionList(self):
        localctx = fugue_sqlParser.QualifiedColTypeWithPositionListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 380, self.RULE_qualifiedColTypeWithPositionList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3629
            self.qualifiedColTypeWithPosition()
            self.state = 3634
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 3630
                self.match(fugue_sqlParser.T__1)
                self.state = 3631
                self.qualifiedColTypeWithPosition()
                self.state = 3636
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
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

        def multipartIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

        def colPosition(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_qualifiedColTypeWithPosition

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedColTypeWithPosition'):
                return visitor.visitQualifiedColTypeWithPosition(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedColTypeWithPosition(self):
        localctx = fugue_sqlParser.QualifiedColTypeWithPositionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 382, self.RULE_qualifiedColTypeWithPosition)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3637
            localctx.name = self.multipartIdentifier()
            self.state = 3638
            self.dataType()
            self.state = 3641
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 199:
                self.state = 3639
                self.match(fugue_sqlParser.NOT)
                self.state = 3640
                self.match(fugue_sqlParser.THENULL)
            self.state = 3644
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 91:
                self.state = 3643
                self.commentSpec()
            self.state = 3647
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 59 or _la == 140:
                self.state = 3646
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
                return self.getTypedRuleContexts(fugue_sqlParser.ColTypeContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ColTypeContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_colTypeList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColTypeList'):
                return visitor.visitColTypeList(self)
            else:
                return visitor.visitChildren(self)

    def colTypeList(self):
        localctx = fugue_sqlParser.ColTypeListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 384, self.RULE_colTypeList)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3649
            self.colType()
            self.state = 3654
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 490, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3650
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3651
                    self.colType()
                self.state = 3656
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 490, self._ctx)
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
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_colType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitColType'):
                return visitor.visitColType(self)
            else:
                return visitor.visitChildren(self)

    def colType(self):
        localctx = fugue_sqlParser.ColTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 386, self.RULE_colType)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3657
            localctx.colName = self.errorCapturingIdentifier()
            self.state = 3658
            self.dataType()
            self.state = 3661
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 491, self._ctx)
            if la_ == 1:
                self.state = 3659
                self.match(fugue_sqlParser.NOT)
                self.state = 3660
                self.match(fugue_sqlParser.THENULL)
            self.state = 3664
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 492, self._ctx)
            if la_ == 1:
                self.state = 3663
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
                return self.getTypedRuleContexts(fugue_sqlParser.ComplexColTypeContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ComplexColTypeContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_complexColTypeList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComplexColTypeList'):
                return visitor.visitComplexColTypeList(self)
            else:
                return visitor.visitChildren(self)

    def complexColTypeList(self):
        localctx = fugue_sqlParser.ComplexColTypeListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 388, self.RULE_complexColTypeList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3666
            self.complexColType()
            self.state = 3671
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 3667
                self.match(fugue_sqlParser.T__1)
                self.state = 3668
                self.complexColType()
                self.state = 3673
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def dataType(self):
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_complexColType

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitComplexColType'):
                return visitor.visitComplexColType(self)
            else:
                return visitor.visitChildren(self)

    def complexColType(self):
        localctx = fugue_sqlParser.ComplexColTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 390, self.RULE_complexColType)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3674
            self.identifier()
            self.state = 3675
            self.match(fugue_sqlParser.T__3)
            self.state = 3676
            self.dataType()
            self.state = 3679
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 199:
                self.state = 3677
                self.match(fugue_sqlParser.NOT)
                self.state = 3678
                self.match(fugue_sqlParser.THENULL)
            self.state = 3682
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la == 91:
                self.state = 3681
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
            return self.getToken(fugue_sqlParser.WHEN, 0)

        def THEN(self):
            return self.getToken(fugue_sqlParser.THEN, 0)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_whenClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWhenClause'):
                return visitor.visitWhenClause(self)
            else:
                return visitor.visitChildren(self)

    def whenClause(self):
        localctx = fugue_sqlParser.WhenClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 392, self.RULE_whenClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3684
            self.match(fugue_sqlParser.WHEN)
            self.state = 3685
            localctx.condition = self.expression()
            self.state = 3686
            self.match(fugue_sqlParser.THEN)
            self.state = 3687
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
            return self.getToken(fugue_sqlParser.WINDOW, 0)

        def namedWindow(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.NamedWindowContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.NamedWindowContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_windowClause

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowClause'):
                return visitor.visitWindowClause(self)
            else:
                return visitor.visitChildren(self)

    def windowClause(self):
        localctx = fugue_sqlParser.WindowClauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 394, self.RULE_windowClause)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3689
            self.match(fugue_sqlParser.WINDOW)
            self.state = 3690
            self.namedWindow()
            self.state = 3695
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 496, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3691
                    self.match(fugue_sqlParser.T__1)
                    self.state = 3692
                    self.namedWindow()
                self.state = 3697
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 496, self._ctx)
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
            return self.getToken(fugue_sqlParser.AS, 0)

        def windowSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowSpecContext, 0)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_namedWindow

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNamedWindow'):
                return visitor.visitNamedWindow(self)
            else:
                return visitor.visitChildren(self)

    def namedWindow(self):
        localctx = fugue_sqlParser.NamedWindowContext(self, self._ctx, self.state)
        self.enterRule(localctx, 396, self.RULE_namedWindow)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3698
            localctx.name = self.errorCapturingIdentifier()
            self.state = 3699
            self.match(fugue_sqlParser.AS)
            self.state = 3700
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
            return fugue_sqlParser.RULE_windowSpec

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class WindowRefContext(WindowSpecContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.name = None
            self.copyFrom(ctx)

        def errorCapturingIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingIdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.CLUSTER, 0)

        def BY(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.BY)
            else:
                return self.getToken(fugue_sqlParser.BY, i)

        def expression(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

        def windowFrame(self):
            return self.getTypedRuleContext(fugue_sqlParser.WindowFrameContext, 0)

        def sortItem(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.SortItemContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.SortItemContext, i)

        def PARTITION(self):
            return self.getToken(fugue_sqlParser.PARTITION, 0)

        def DISTRIBUTE(self):
            return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

        def ORDER(self):
            return self.getToken(fugue_sqlParser.ORDER, 0)

        def SORT(self):
            return self.getToken(fugue_sqlParser.SORT, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowDef'):
                return visitor.visitWindowDef(self)
            else:
                return visitor.visitChildren(self)

    def windowSpec(self):
        localctx = fugue_sqlParser.WindowSpecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 398, self.RULE_windowSpec)
        self._la = 0
        try:
            self.state = 3748
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 504, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.WindowRefContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3702
                localctx.name = self.errorCapturingIdentifier()
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.WindowRefContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3703
                self.match(fugue_sqlParser.T__4)
                self.state = 3704
                localctx.name = self.errorCapturingIdentifier()
                self.state = 3705
                self.match(fugue_sqlParser.T__5)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.WindowDefContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 3707
                self.match(fugue_sqlParser.T__4)
                self.state = 3742
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [84]:
                    self.state = 3708
                    self.match(fugue_sqlParser.CLUSTER)
                    self.state = 3709
                    self.match(fugue_sqlParser.BY)
                    self.state = 3710
                    localctx._expression = self.expression()
                    localctx.partition.append(localctx._expression)
                    self.state = 3715
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la == 2:
                        self.state = 3711
                        self.match(fugue_sqlParser.T__1)
                        self.state = 3712
                        localctx._expression = self.expression()
                        localctx.partition.append(localctx._expression)
                        self.state = 3717
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                    pass
                elif token in [6, 121, 208, 216, 229, 249, 264]:
                    self.state = 3728
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 121 or _la == 216:
                        self.state = 3718
                        _la = self._input.LA(1)
                        if not (_la == 121 or _la == 216):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 3719
                        self.match(fugue_sqlParser.BY)
                        self.state = 3720
                        localctx._expression = self.expression()
                        localctx.partition.append(localctx._expression)
                        self.state = 3725
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        while _la == 2:
                            self.state = 3721
                            self.match(fugue_sqlParser.T__1)
                            self.state = 3722
                            localctx._expression = self.expression()
                            localctx.partition.append(localctx._expression)
                            self.state = 3727
                            self._errHandler.sync(self)
                            _la = self._input.LA(1)
                    self.state = 3740
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la == 208 or _la == 264:
                        self.state = 3730
                        _la = self._input.LA(1)
                        if not (_la == 208 or _la == 264):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 3731
                        self.match(fugue_sqlParser.BY)
                        self.state = 3732
                        self.sortItem()
                        self.state = 3737
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)
                        while _la == 2:
                            self.state = 3733
                            self.match(fugue_sqlParser.T__1)
                            self.state = 3734
                            self.sortItem()
                            self.state = 3739
                            self._errHandler.sync(self)
                            _la = self._input.LA(1)
                    pass
                else:
                    raise NoViableAltException(self)
                self.state = 3745
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 229 or _la == 249:
                    self.state = 3744
                    self.windowFrame()
                self.state = 3747
                self.match(fugue_sqlParser.T__5)
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
            return self.getToken(fugue_sqlParser.RANGE, 0)

        def frameBound(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.FrameBoundContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.FrameBoundContext, i)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def BETWEEN(self):
            return self.getToken(fugue_sqlParser.BETWEEN, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_windowFrame

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitWindowFrame'):
                return visitor.visitWindowFrame(self)
            else:
                return visitor.visitChildren(self)

    def windowFrame(self):
        localctx = fugue_sqlParser.WindowFrameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 400, self.RULE_windowFrame)
        try:
            self.state = 3766
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 505, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3750
                localctx.frameType = self.match(fugue_sqlParser.RANGE)
                self.state = 3751
                localctx.start = self.frameBound()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3752
                localctx.frameType = self.match(fugue_sqlParser.ROWS)
                self.state = 3753
                localctx.start = self.frameBound()
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 3754
                localctx.frameType = self.match(fugue_sqlParser.RANGE)
                self.state = 3755
                self.match(fugue_sqlParser.BETWEEN)
                self.state = 3756
                localctx.start = self.frameBound()
                self.state = 3757
                self.match(fugue_sqlParser.AND)
                self.state = 3758
                localctx.end = self.frameBound()
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 3760
                localctx.frameType = self.match(fugue_sqlParser.ROWS)
                self.state = 3761
                self.match(fugue_sqlParser.BETWEEN)
                self.state = 3762
                localctx.start = self.frameBound()
                self.state = 3763
                self.match(fugue_sqlParser.AND)
                self.state = 3764
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
            return self.getToken(fugue_sqlParser.UNBOUNDED, 0)

        def PRECEDING(self):
            return self.getToken(fugue_sqlParser.PRECEDING, 0)

        def FOLLOWING(self):
            return self.getToken(fugue_sqlParser.FOLLOWING, 0)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def CURRENT(self):
            return self.getToken(fugue_sqlParser.CURRENT, 0)

        def expression(self):
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_frameBound

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFrameBound'):
                return visitor.visitFrameBound(self)
            else:
                return visitor.visitChildren(self)

    def frameBound(self):
        localctx = fugue_sqlParser.FrameBoundContext(self, self._ctx, self.state)
        self.enterRule(localctx, 402, self.RULE_frameBound)
        self._la = 0
        try:
            self.state = 3775
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 506, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3768
                self.match(fugue_sqlParser.UNBOUNDED)
                self.state = 3769
                localctx.boundType = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 141 or _la == 223):
                    localctx.boundType = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3770
                localctx.boundType = self.match(fugue_sqlParser.CURRENT)
                self.state = 3771
                self.match(fugue_sqlParser.ROW)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 3772
                self.expression()
                self.state = 3773
                localctx.boundType = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 141 or _la == 223):
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
                return self.getTypedRuleContexts(fugue_sqlParser.QualifiedNameContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_qualifiedNameList

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedNameList'):
                return visitor.visitQualifiedNameList(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedNameList(self):
        localctx = fugue_sqlParser.QualifiedNameListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 404, self.RULE_qualifiedNameList)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3777
            self.qualifiedName()
            self.state = 3782
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la == 2:
                self.state = 3778
                self.match(fugue_sqlParser.T__1)
                self.state = 3779
                self.qualifiedName()
                self.state = 3784
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
            return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

        def FILTER(self):
            return self.getToken(fugue_sqlParser.FILTER, 0)

        def LEFT(self):
            return self.getToken(fugue_sqlParser.LEFT, 0)

        def RIGHT(self):
            return self.getToken(fugue_sqlParser.RIGHT, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_functionName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitFunctionName'):
                return visitor.visitFunctionName(self)
            else:
                return visitor.visitChildren(self)

    def functionName(self):
        localctx = fugue_sqlParser.FunctionNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 406, self.RULE_functionName)
        try:
            self.state = 3789
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 508, self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 3785
                self.qualifiedName()
                pass
            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 3786
                self.match(fugue_sqlParser.FILTER)
                pass
            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 3787
                self.match(fugue_sqlParser.LEFT)
                pass
            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 3788
                self.match(fugue_sqlParser.RIGHT)
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
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_qualifiedName

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQualifiedName'):
                return visitor.visitQualifiedName(self)
            else:
                return visitor.visitChildren(self)

    def qualifiedName(self):
        localctx = fugue_sqlParser.QualifiedNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 408, self.RULE_qualifiedName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3791
            self.identifier()
            self.state = 3796
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input, 509, self._ctx)
            while _alt != 2 and _alt != ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 3792
                    self.match(fugue_sqlParser.T__6)
                    self.state = 3793
                    self.identifier()
                self.state = 3798
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input, 509, self._ctx)
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
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_errorCapturingIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorCapturingIdentifier'):
                return visitor.visitErrorCapturingIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingIdentifier(self):
        localctx = fugue_sqlParser.ErrorCapturingIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 410, self.RULE_errorCapturingIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3799
            self.identifier()
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
            return fugue_sqlParser.RULE_errorCapturingIdentifierExtra

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class ErrorIdentContext(ErrorCapturingIdentifierExtraContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def MINUS(self, i: int=None):
            if i is None:
                return self.getTokens(fugue_sqlParser.MINUS)
            else:
                return self.getToken(fugue_sqlParser.MINUS, i)

        def identifier(self, i: int=None):
            if i is None:
                return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
            else:
                return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitErrorIdent'):
                return visitor.visitErrorIdent(self)
            else:
                return visitor.visitChildren(self)

    def errorCapturingIdentifierExtra(self):
        localctx = fugue_sqlParser.ErrorCapturingIdentifierExtraContext(self, self._ctx, self.state)
        self.enterRule(localctx, 412, self.RULE_errorCapturingIdentifierExtra)
        self._la = 0
        try:
            localctx = fugue_sqlParser.ErrorIdentContext(self, localctx)
            self.enterOuterAlt(localctx, 1)
            self.state = 3803
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 3801
                self.match(fugue_sqlParser.MINUS)
                self.state = 3802
                self.identifier()
                self.state = 3805
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not _la == 320:
                    break
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
            return self.getTypedRuleContext(fugue_sqlParser.StrictIdentifierContext, 0)

        def strictNonReserved(self):
            return self.getTypedRuleContext(fugue_sqlParser.StrictNonReservedContext, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_identifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitIdentifier'):
                return visitor.visitIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def identifier(self):
        localctx = fugue_sqlParser.IdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 414, self.RULE_identifier)
        try:
            self.state = 3809
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 167, 168, 169, 170, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 324, 339, 340]:
                self.enterOuterAlt(localctx, 1)
                self.state = 3807
                self.strictIdentifier()
                pass
            elif token in [64, 100, 127, 147, 162, 166, 171, 177, 197, 203, 242, 253, 259, 293, 301]:
                self.enterOuterAlt(localctx, 2)
                self.state = 3808
                self.strictNonReserved()
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

    class StrictIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_strictIdentifier

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class QuotedIdentifierAlternativeContext(StrictIdentifierContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def quotedIdentifier(self):
            return self.getTypedRuleContext(fugue_sqlParser.QuotedIdentifierContext, 0)

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
            return self.getToken(fugue_sqlParser.IDENTIFIER, 0)

        def nonReserved(self):
            return self.getTypedRuleContext(fugue_sqlParser.NonReservedContext, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitUnquotedIdentifier'):
                return visitor.visitUnquotedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def strictIdentifier(self):
        localctx = fugue_sqlParser.StrictIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 416, self.RULE_strictIdentifier)
        try:
            self.state = 3814
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [339]:
                localctx = fugue_sqlParser.UnquotedIdentifierContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3811
                self.match(fugue_sqlParser.IDENTIFIER)
                pass
            elif token in [340]:
                localctx = fugue_sqlParser.QuotedIdentifierAlternativeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3812
                self.quotedIdentifier()
                pass
            elif token in [58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 167, 168, 169, 170, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 324]:
                localctx = fugue_sqlParser.UnquotedIdentifierContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 3813
                self.nonReserved()
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

    class QuotedIdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BACKQUOTED_IDENTIFIER(self):
            return self.getToken(fugue_sqlParser.BACKQUOTED_IDENTIFIER, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_quotedIdentifier

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitQuotedIdentifier'):
                return visitor.visitQuotedIdentifier(self)
            else:
                return visitor.visitChildren(self)

    def quotedIdentifier(self):
        localctx = fugue_sqlParser.QuotedIdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 418, self.RULE_quotedIdentifier)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3816
            self.match(fugue_sqlParser.BACKQUOTED_IDENTIFIER)
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
            return fugue_sqlParser.RULE_number

        def copyFrom(self, ctx: ParserRuleContext):
            super().copyFrom(ctx)

    class DecimalLiteralContext(NumberContext):

        def __init__(self, parser, ctx: ParserRuleContext):
            super().__init__(parser)
            self.copyFrom(ctx)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.BIGINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.TINYINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.EXPONENT_VALUE, 0)

        def DECIMAL_VALUE(self):
            return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.BIGDECIMAL_LITERAL, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.EXPONENT_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.DOUBLE_LITERAL, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

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
            return self.getToken(fugue_sqlParser.SMALLINT_LITERAL, 0)

        def MINUS(self):
            return self.getToken(fugue_sqlParser.MINUS, 0)

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitSmallIntLiteral'):
                return visitor.visitSmallIntLiteral(self)
            else:
                return visitor.visitChildren(self)

    def number(self):
        localctx = fugue_sqlParser.NumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 420, self.RULE_number)
        self._la = 0
        try:
            self.state = 3854
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input, 522, self._ctx)
            if la_ == 1:
                localctx = fugue_sqlParser.ExponentLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 3819
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3818
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3821
                self.match(fugue_sqlParser.EXPONENT_VALUE)
                pass
            elif la_ == 2:
                localctx = fugue_sqlParser.DecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 3823
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3822
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3825
                self.match(fugue_sqlParser.DECIMAL_VALUE)
                pass
            elif la_ == 3:
                localctx = fugue_sqlParser.LegacyDecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 3827
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3826
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3829
                _la = self._input.LA(1)
                if not (_la == 335 or _la == 336):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif la_ == 4:
                localctx = fugue_sqlParser.IntegerLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 3831
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3830
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3833
                self.match(fugue_sqlParser.INTEGER_VALUE)
                pass
            elif la_ == 5:
                localctx = fugue_sqlParser.BigIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 3835
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3834
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3837
                self.match(fugue_sqlParser.BIGINT_LITERAL)
                pass
            elif la_ == 6:
                localctx = fugue_sqlParser.SmallIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 3839
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3838
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3841
                self.match(fugue_sqlParser.SMALLINT_LITERAL)
                pass
            elif la_ == 7:
                localctx = fugue_sqlParser.TinyIntLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 3843
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3842
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3845
                self.match(fugue_sqlParser.TINYINT_LITERAL)
                pass
            elif la_ == 8:
                localctx = fugue_sqlParser.DoubleLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 3847
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3846
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3849
                self.match(fugue_sqlParser.DOUBLE_LITERAL)
                pass
            elif la_ == 9:
                localctx = fugue_sqlParser.BigDecimalLiteralContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 3851
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la == 320:
                    self.state = 3850
                    self.match(fugue_sqlParser.MINUS)
                self.state = 3853
                self.match(fugue_sqlParser.BIGDECIMAL_LITERAL)
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
            return self.getToken(fugue_sqlParser.TYPE, 0)

        def dataType(self):
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, 0)

        def commentSpec(self):
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)

        def colPosition(self):
            return self.getTypedRuleContext(fugue_sqlParser.ColPositionContext, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_alterColumnAction

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAlterColumnAction'):
                return visitor.visitAlterColumnAction(self)
            else:
                return visitor.visitChildren(self)

    def alterColumnAction(self):
        localctx = fugue_sqlParser.AlterColumnActionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 422, self.RULE_alterColumnAction)
        self._la = 0
        try:
            self.state = 3863
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [289]:
                self.enterOuterAlt(localctx, 1)
                self.state = 3856
                self.match(fugue_sqlParser.TYPE)
                self.state = 3857
                self.dataType()
                pass
            elif token in [91]:
                self.enterOuterAlt(localctx, 2)
                self.state = 3858
                self.commentSpec()
                pass
            elif token in [59, 140]:
                self.enterOuterAlt(localctx, 3)
                self.state = 3859
                self.colPosition()
                pass
            elif token in [122, 258]:
                self.enterOuterAlt(localctx, 4)
                self.state = 3860
                localctx.setOrDrop = self._input.LT(1)
                _la = self._input.LA(1)
                if not (_la == 122 or _la == 258):
                    localctx.setOrDrop = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 3861
                self.match(fugue_sqlParser.NOT)
                self.state = 3862
                self.match(fugue_sqlParser.THENULL)
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
            return self.getToken(fugue_sqlParser.ADD, 0)

        def AFTER(self):
            return self.getToken(fugue_sqlParser.AFTER, 0)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def ANALYZE(self):
            return self.getToken(fugue_sqlParser.ANALYZE, 0)

        def ARCHIVE(self):
            return self.getToken(fugue_sqlParser.ARCHIVE, 0)

        def ARRAY(self):
            return self.getToken(fugue_sqlParser.ARRAY, 0)

        def ASC(self):
            return self.getToken(fugue_sqlParser.ASC, 0)

        def AT(self):
            return self.getToken(fugue_sqlParser.AT, 0)

        def BETWEEN(self):
            return self.getToken(fugue_sqlParser.BETWEEN, 0)

        def BUCKET(self):
            return self.getToken(fugue_sqlParser.BUCKET, 0)

        def BUCKETS(self):
            return self.getToken(fugue_sqlParser.BUCKETS, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def CACHE(self):
            return self.getToken(fugue_sqlParser.CACHE, 0)

        def CASCADE(self):
            return self.getToken(fugue_sqlParser.CASCADE, 0)

        def CHANGE(self):
            return self.getToken(fugue_sqlParser.CHANGE, 0)

        def CLEAR(self):
            return self.getToken(fugue_sqlParser.CLEAR, 0)

        def CLUSTER(self):
            return self.getToken(fugue_sqlParser.CLUSTER, 0)

        def CLUSTERED(self):
            return self.getToken(fugue_sqlParser.CLUSTERED, 0)

        def CODEGEN(self):
            return self.getToken(fugue_sqlParser.CODEGEN, 0)

        def COLLECTION(self):
            return self.getToken(fugue_sqlParser.COLLECTION, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def COMMENT(self):
            return self.getToken(fugue_sqlParser.COMMENT, 0)

        def COMMIT(self):
            return self.getToken(fugue_sqlParser.COMMIT, 0)

        def COMPACT(self):
            return self.getToken(fugue_sqlParser.COMPACT, 0)

        def COMPACTIONS(self):
            return self.getToken(fugue_sqlParser.COMPACTIONS, 0)

        def COMPUTE(self):
            return self.getToken(fugue_sqlParser.COMPUTE, 0)

        def CONCATENATE(self):
            return self.getToken(fugue_sqlParser.CONCATENATE, 0)

        def COST(self):
            return self.getToken(fugue_sqlParser.COST, 0)

        def CUBE(self):
            return self.getToken(fugue_sqlParser.CUBE, 0)

        def CURRENT(self):
            return self.getToken(fugue_sqlParser.CURRENT, 0)

        def DATA(self):
            return self.getToken(fugue_sqlParser.DATA, 0)

        def DATABASE(self):
            return self.getToken(fugue_sqlParser.DATABASE, 0)

        def DATABASES(self):
            return self.getToken(fugue_sqlParser.DATABASES, 0)

        def DBPROPERTIES(self):
            return self.getToken(fugue_sqlParser.DBPROPERTIES, 0)

        def DEFINED(self):
            return self.getToken(fugue_sqlParser.DEFINED, 0)

        def DELETE(self):
            return self.getToken(fugue_sqlParser.DELETE, 0)

        def DELIMITED(self):
            return self.getToken(fugue_sqlParser.DELIMITED, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def DFS(self):
            return self.getToken(fugue_sqlParser.DFS, 0)

        def DIRECTORIES(self):
            return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

        def DIRECTORY(self):
            return self.getToken(fugue_sqlParser.DIRECTORY, 0)

        def DISTRIBUTE(self):
            return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

        def DIV(self):
            return self.getToken(fugue_sqlParser.DIV, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def ESCAPED(self):
            return self.getToken(fugue_sqlParser.ESCAPED, 0)

        def EXCHANGE(self):
            return self.getToken(fugue_sqlParser.EXCHANGE, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def EXPLAIN(self):
            return self.getToken(fugue_sqlParser.EXPLAIN, 0)

        def EXPORT(self):
            return self.getToken(fugue_sqlParser.EXPORT, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def EXTERNAL(self):
            return self.getToken(fugue_sqlParser.EXTERNAL, 0)

        def EXTRACT(self):
            return self.getToken(fugue_sqlParser.EXTRACT, 0)

        def FIELDS(self):
            return self.getToken(fugue_sqlParser.FIELDS, 0)

        def FILEFORMAT(self):
            return self.getToken(fugue_sqlParser.FILEFORMAT, 0)

        def FIRST(self):
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def FOLLOWING(self):
            return self.getToken(fugue_sqlParser.FOLLOWING, 0)

        def FORMAT(self):
            return self.getToken(fugue_sqlParser.FORMAT, 0)

        def FORMATTED(self):
            return self.getToken(fugue_sqlParser.FORMATTED, 0)

        def FUNCTION(self):
            return self.getToken(fugue_sqlParser.FUNCTION, 0)

        def FUNCTIONS(self):
            return self.getToken(fugue_sqlParser.FUNCTIONS, 0)

        def GLOBAL(self):
            return self.getToken(fugue_sqlParser.GLOBAL, 0)

        def GROUPING(self):
            return self.getToken(fugue_sqlParser.GROUPING, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def IGNORE(self):
            return self.getToken(fugue_sqlParser.IGNORE, 0)

        def IMPORT(self):
            return self.getToken(fugue_sqlParser.IMPORT, 0)

        def INDEX(self):
            return self.getToken(fugue_sqlParser.INDEX, 0)

        def INDEXES(self):
            return self.getToken(fugue_sqlParser.INDEXES, 0)

        def INPATH(self):
            return self.getToken(fugue_sqlParser.INPATH, 0)

        def INPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)

        def INSERT(self):
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def INTERVAL(self):
            return self.getToken(fugue_sqlParser.INTERVAL, 0)

        def ITEMS(self):
            return self.getToken(fugue_sqlParser.ITEMS, 0)

        def KEYS(self):
            return self.getToken(fugue_sqlParser.KEYS, 0)

        def LAST(self):
            return self.getToken(fugue_sqlParser.LAST, 0)

        def LATERAL(self):
            return self.getToken(fugue_sqlParser.LATERAL, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def LIMIT(self):
            return self.getToken(fugue_sqlParser.LIMIT, 0)

        def LINES(self):
            return self.getToken(fugue_sqlParser.LINES, 0)

        def LIST(self):
            return self.getToken(fugue_sqlParser.LIST, 0)

        def LOAD(self):
            return self.getToken(fugue_sqlParser.LOAD, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def LOCATION(self):
            return self.getToken(fugue_sqlParser.LOCATION, 0)

        def LOCK(self):
            return self.getToken(fugue_sqlParser.LOCK, 0)

        def LOCKS(self):
            return self.getToken(fugue_sqlParser.LOCKS, 0)

        def LOGICAL(self):
            return self.getToken(fugue_sqlParser.LOGICAL, 0)

        def MACRO(self):
            return self.getToken(fugue_sqlParser.MACRO, 0)

        def MAP(self):
            return self.getToken(fugue_sqlParser.MAP, 0)

        def MATCHED(self):
            return self.getToken(fugue_sqlParser.MATCHED, 0)

        def MERGE(self):
            return self.getToken(fugue_sqlParser.MERGE, 0)

        def MSCK(self):
            return self.getToken(fugue_sqlParser.MSCK, 0)

        def NAMESPACE(self):
            return self.getToken(fugue_sqlParser.NAMESPACE, 0)

        def NAMESPACES(self):
            return self.getToken(fugue_sqlParser.NAMESPACES, 0)

        def NO(self):
            return self.getToken(fugue_sqlParser.NO, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def OF(self):
            return self.getToken(fugue_sqlParser.OF, 0)

        def OPTION(self):
            return self.getToken(fugue_sqlParser.OPTION, 0)

        def OPTIONS(self):
            return self.getToken(fugue_sqlParser.OPTIONS, 0)

        def OUT(self):
            return self.getToken(fugue_sqlParser.OUT, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)

        def OVER(self):
            return self.getToken(fugue_sqlParser.OVER, 0)

        def OVERLAY(self):
            return self.getToken(fugue_sqlParser.OVERLAY, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def PARTITION(self):
            return self.getToken(fugue_sqlParser.PARTITION, 0)

        def PARTITIONED(self):
            return self.getToken(fugue_sqlParser.PARTITIONED, 0)

        def PARTITIONS(self):
            return self.getToken(fugue_sqlParser.PARTITIONS, 0)

        def PERCENTLIT(self):
            return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

        def PIVOT(self):
            return self.getToken(fugue_sqlParser.PIVOT, 0)

        def PLACING(self):
            return self.getToken(fugue_sqlParser.PLACING, 0)

        def POSITION(self):
            return self.getToken(fugue_sqlParser.POSITION, 0)

        def PRECEDING(self):
            return self.getToken(fugue_sqlParser.PRECEDING, 0)

        def PRINCIPALS(self):
            return self.getToken(fugue_sqlParser.PRINCIPALS, 0)

        def PROPERTIES(self):
            return self.getToken(fugue_sqlParser.PROPERTIES, 0)

        def PURGE(self):
            return self.getToken(fugue_sqlParser.PURGE, 0)

        def QUERY(self):
            return self.getToken(fugue_sqlParser.QUERY, 0)

        def RANGE(self):
            return self.getToken(fugue_sqlParser.RANGE, 0)

        def RECORDREADER(self):
            return self.getToken(fugue_sqlParser.RECORDREADER, 0)

        def RECORDWRITER(self):
            return self.getToken(fugue_sqlParser.RECORDWRITER, 0)

        def RECOVER(self):
            return self.getToken(fugue_sqlParser.RECOVER, 0)

        def REDUCE(self):
            return self.getToken(fugue_sqlParser.REDUCE, 0)

        def REFRESH(self):
            return self.getToken(fugue_sqlParser.REFRESH, 0)

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def REPAIR(self):
            return self.getToken(fugue_sqlParser.REPAIR, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def RESET(self):
            return self.getToken(fugue_sqlParser.RESET, 0)

        def RESTRICT(self):
            return self.getToken(fugue_sqlParser.RESTRICT, 0)

        def REVOKE(self):
            return self.getToken(fugue_sqlParser.REVOKE, 0)

        def RLIKE(self):
            return self.getToken(fugue_sqlParser.RLIKE, 0)

        def ROLE(self):
            return self.getToken(fugue_sqlParser.ROLE, 0)

        def ROLES(self):
            return self.getToken(fugue_sqlParser.ROLES, 0)

        def ROLLBACK(self):
            return self.getToken(fugue_sqlParser.ROLLBACK, 0)

        def ROLLUP(self):
            return self.getToken(fugue_sqlParser.ROLLUP, 0)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def SEPARATED(self):
            return self.getToken(fugue_sqlParser.SEPARATED, 0)

        def SERDE(self):
            return self.getToken(fugue_sqlParser.SERDE, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def SETS(self):
            return self.getToken(fugue_sqlParser.SETS, 0)

        def SHOW(self):
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def SKEWED(self):
            return self.getToken(fugue_sqlParser.SKEWED, 0)

        def SORT(self):
            return self.getToken(fugue_sqlParser.SORT, 0)

        def SORTED(self):
            return self.getToken(fugue_sqlParser.SORTED, 0)

        def START(self):
            return self.getToken(fugue_sqlParser.START, 0)

        def STATISTICS(self):
            return self.getToken(fugue_sqlParser.STATISTICS, 0)

        def STORED(self):
            return self.getToken(fugue_sqlParser.STORED, 0)

        def STRATIFY(self):
            return self.getToken(fugue_sqlParser.STRATIFY, 0)

        def STRUCT(self):
            return self.getToken(fugue_sqlParser.STRUCT, 0)

        def SUBSTR(self):
            return self.getToken(fugue_sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(fugue_sqlParser.SUBSTRING, 0)

        def TABLES(self):
            return self.getToken(fugue_sqlParser.TABLES, 0)

        def TABLESAMPLE(self):
            return self.getToken(fugue_sqlParser.TABLESAMPLE, 0)

        def TBLPROPERTIES(self):
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def TERMINATED(self):
            return self.getToken(fugue_sqlParser.TERMINATED, 0)

        def TOUCH(self):
            return self.getToken(fugue_sqlParser.TOUCH, 0)

        def TRANSACTION(self):
            return self.getToken(fugue_sqlParser.TRANSACTION, 0)

        def TRANSACTIONS(self):
            return self.getToken(fugue_sqlParser.TRANSACTIONS, 0)

        def TRANSFORM(self):
            return self.getToken(fugue_sqlParser.TRANSFORM, 0)

        def TRIM(self):
            return self.getToken(fugue_sqlParser.TRIM, 0)

        def TRUE(self):
            return self.getToken(fugue_sqlParser.TRUE, 0)

        def TRUNCATE(self):
            return self.getToken(fugue_sqlParser.TRUNCATE, 0)

        def UNARCHIVE(self):
            return self.getToken(fugue_sqlParser.UNARCHIVE, 0)

        def UNBOUNDED(self):
            return self.getToken(fugue_sqlParser.UNBOUNDED, 0)

        def UNCACHE(self):
            return self.getToken(fugue_sqlParser.UNCACHE, 0)

        def UNLOCK(self):
            return self.getToken(fugue_sqlParser.UNLOCK, 0)

        def UNSET(self):
            return self.getToken(fugue_sqlParser.UNSET, 0)

        def UPDATE(self):
            return self.getToken(fugue_sqlParser.UPDATE, 0)

        def USE(self):
            return self.getToken(fugue_sqlParser.USE, 0)

        def VALUES(self):
            return self.getToken(fugue_sqlParser.VALUES, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def VIEWS(self):
            return self.getToken(fugue_sqlParser.VIEWS, 0)

        def WINDOW(self):
            return self.getToken(fugue_sqlParser.WINDOW, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_ansiNonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitAnsiNonReserved'):
                return visitor.visitAnsiNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def ansiNonReserved(self):
        localctx = fugue_sqlParser.AnsiNonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 424, self.RULE_ansiNonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3865
            _la = self._input.LA(1)
            if not (_la - 58 & ~63 == 0 and 1 << _la - 58 & -4616724533169136869 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -54836095400108079 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -72339344050251969 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 176704157053345137 != 0) or (_la == 324)):
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
            return self.getToken(fugue_sqlParser.ANTI, 0)

        def CROSS(self):
            return self.getToken(fugue_sqlParser.CROSS, 0)

        def EXCEPT(self):
            return self.getToken(fugue_sqlParser.EXCEPT, 0)

        def FULL(self):
            return self.getToken(fugue_sqlParser.FULL, 0)

        def INNER(self):
            return self.getToken(fugue_sqlParser.INNER, 0)

        def INTERSECT(self):
            return self.getToken(fugue_sqlParser.INTERSECT, 0)

        def JOIN(self):
            return self.getToken(fugue_sqlParser.JOIN, 0)

        def LEFT(self):
            return self.getToken(fugue_sqlParser.LEFT, 0)

        def NATURAL(self):
            return self.getToken(fugue_sqlParser.NATURAL, 0)

        def ON(self):
            return self.getToken(fugue_sqlParser.ON, 0)

        def RIGHT(self):
            return self.getToken(fugue_sqlParser.RIGHT, 0)

        def SEMI(self):
            return self.getToken(fugue_sqlParser.SEMI, 0)

        def SETMINUS(self):
            return self.getToken(fugue_sqlParser.SETMINUS, 0)

        def UNION(self):
            return self.getToken(fugue_sqlParser.UNION, 0)

        def USING(self):
            return self.getToken(fugue_sqlParser.USING, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_strictNonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitStrictNonReserved'):
                return visitor.visitStrictNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def strictNonReserved(self):
        localctx = fugue_sqlParser.StrictNonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 426, self.RULE_strictNonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3867
            _la = self._input.LA(1)
            if not (_la - 64 & ~63 == 0 and 1 << _la - 64 & -9223371968135299071 != 0 or (_la - 147 & ~63 == 0 and 1 << _la - 147 & 73183495035846657 != 0) or (_la - 242 & ~63 == 0 and 1 << _la - 242 & 578712552117241857 != 0)):
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
            return self.getToken(fugue_sqlParser.ADD, 0)

        def AFTER(self):
            return self.getToken(fugue_sqlParser.AFTER, 0)

        def ALL(self):
            return self.getToken(fugue_sqlParser.ALL, 0)

        def ALTER(self):
            return self.getToken(fugue_sqlParser.ALTER, 0)

        def ANALYZE(self):
            return self.getToken(fugue_sqlParser.ANALYZE, 0)

        def AND(self):
            return self.getToken(fugue_sqlParser.AND, 0)

        def ANY(self):
            return self.getToken(fugue_sqlParser.ANY, 0)

        def ARCHIVE(self):
            return self.getToken(fugue_sqlParser.ARCHIVE, 0)

        def ARRAY(self):
            return self.getToken(fugue_sqlParser.ARRAY, 0)

        def AS(self):
            return self.getToken(fugue_sqlParser.AS, 0)

        def ASC(self):
            return self.getToken(fugue_sqlParser.ASC, 0)

        def AT(self):
            return self.getToken(fugue_sqlParser.AT, 0)

        def AUTHORIZATION(self):
            return self.getToken(fugue_sqlParser.AUTHORIZATION, 0)

        def BETWEEN(self):
            return self.getToken(fugue_sqlParser.BETWEEN, 0)

        def BOTH(self):
            return self.getToken(fugue_sqlParser.BOTH, 0)

        def BUCKET(self):
            return self.getToken(fugue_sqlParser.BUCKET, 0)

        def BUCKETS(self):
            return self.getToken(fugue_sqlParser.BUCKETS, 0)

        def BY(self):
            return self.getToken(fugue_sqlParser.BY, 0)

        def CACHE(self):
            return self.getToken(fugue_sqlParser.CACHE, 0)

        def CASCADE(self):
            return self.getToken(fugue_sqlParser.CASCADE, 0)

        def CASE(self):
            return self.getToken(fugue_sqlParser.CASE, 0)

        def CAST(self):
            return self.getToken(fugue_sqlParser.CAST, 0)

        def CHANGE(self):
            return self.getToken(fugue_sqlParser.CHANGE, 0)

        def CHECK(self):
            return self.getToken(fugue_sqlParser.CHECK, 0)

        def CLEAR(self):
            return self.getToken(fugue_sqlParser.CLEAR, 0)

        def CLUSTER(self):
            return self.getToken(fugue_sqlParser.CLUSTER, 0)

        def CLUSTERED(self):
            return self.getToken(fugue_sqlParser.CLUSTERED, 0)

        def CODEGEN(self):
            return self.getToken(fugue_sqlParser.CODEGEN, 0)

        def COLLATE(self):
            return self.getToken(fugue_sqlParser.COLLATE, 0)

        def COLLECTION(self):
            return self.getToken(fugue_sqlParser.COLLECTION, 0)

        def COLUMN(self):
            return self.getToken(fugue_sqlParser.COLUMN, 0)

        def COLUMNS(self):
            return self.getToken(fugue_sqlParser.COLUMNS, 0)

        def COMMENT(self):
            return self.getToken(fugue_sqlParser.COMMENT, 0)

        def COMMIT(self):
            return self.getToken(fugue_sqlParser.COMMIT, 0)

        def COMPACT(self):
            return self.getToken(fugue_sqlParser.COMPACT, 0)

        def COMPACTIONS(self):
            return self.getToken(fugue_sqlParser.COMPACTIONS, 0)

        def COMPUTE(self):
            return self.getToken(fugue_sqlParser.COMPUTE, 0)

        def CONCATENATE(self):
            return self.getToken(fugue_sqlParser.CONCATENATE, 0)

        def CONSTRAINT(self):
            return self.getToken(fugue_sqlParser.CONSTRAINT, 0)

        def COST(self):
            return self.getToken(fugue_sqlParser.COST, 0)

        def CREATE(self):
            return self.getToken(fugue_sqlParser.CREATE, 0)

        def CUBE(self):
            return self.getToken(fugue_sqlParser.CUBE, 0)

        def CURRENT(self):
            return self.getToken(fugue_sqlParser.CURRENT, 0)

        def CURRENT_DATE(self):
            return self.getToken(fugue_sqlParser.CURRENT_DATE, 0)

        def CURRENT_TIME(self):
            return self.getToken(fugue_sqlParser.CURRENT_TIME, 0)

        def CURRENT_TIMESTAMP(self):
            return self.getToken(fugue_sqlParser.CURRENT_TIMESTAMP, 0)

        def CURRENT_USER(self):
            return self.getToken(fugue_sqlParser.CURRENT_USER, 0)

        def DATA(self):
            return self.getToken(fugue_sqlParser.DATA, 0)

        def DATABASE(self):
            return self.getToken(fugue_sqlParser.DATABASE, 0)

        def DATABASES(self):
            return self.getToken(fugue_sqlParser.DATABASES, 0)

        def DAY(self):
            return self.getToken(fugue_sqlParser.DAY, 0)

        def DBPROPERTIES(self):
            return self.getToken(fugue_sqlParser.DBPROPERTIES, 0)

        def DEFINED(self):
            return self.getToken(fugue_sqlParser.DEFINED, 0)

        def DELETE(self):
            return self.getToken(fugue_sqlParser.DELETE, 0)

        def DELIMITED(self):
            return self.getToken(fugue_sqlParser.DELIMITED, 0)

        def DESC(self):
            return self.getToken(fugue_sqlParser.DESC, 0)

        def DESCRIBE(self):
            return self.getToken(fugue_sqlParser.DESCRIBE, 0)

        def DFS(self):
            return self.getToken(fugue_sqlParser.DFS, 0)

        def DIRECTORIES(self):
            return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

        def DIRECTORY(self):
            return self.getToken(fugue_sqlParser.DIRECTORY, 0)

        def DISTINCT(self):
            return self.getToken(fugue_sqlParser.DISTINCT, 0)

        def DISTRIBUTE(self):
            return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

        def DIV(self):
            return self.getToken(fugue_sqlParser.DIV, 0)

        def DROP(self):
            return self.getToken(fugue_sqlParser.DROP, 0)

        def ELSE(self):
            return self.getToken(fugue_sqlParser.ELSE, 0)

        def END(self):
            return self.getToken(fugue_sqlParser.END, 0)

        def ESCAPE(self):
            return self.getToken(fugue_sqlParser.ESCAPE, 0)

        def ESCAPED(self):
            return self.getToken(fugue_sqlParser.ESCAPED, 0)

        def EXCHANGE(self):
            return self.getToken(fugue_sqlParser.EXCHANGE, 0)

        def EXISTS(self):
            return self.getToken(fugue_sqlParser.EXISTS, 0)

        def EXPLAIN(self):
            return self.getToken(fugue_sqlParser.EXPLAIN, 0)

        def EXPORT(self):
            return self.getToken(fugue_sqlParser.EXPORT, 0)

        def EXTENDED(self):
            return self.getToken(fugue_sqlParser.EXTENDED, 0)

        def EXTERNAL(self):
            return self.getToken(fugue_sqlParser.EXTERNAL, 0)

        def EXTRACT(self):
            return self.getToken(fugue_sqlParser.EXTRACT, 0)

        def FALSE(self):
            return self.getToken(fugue_sqlParser.FALSE, 0)

        def FETCH(self):
            return self.getToken(fugue_sqlParser.FETCH, 0)

        def FILTER(self):
            return self.getToken(fugue_sqlParser.FILTER, 0)

        def FIELDS(self):
            return self.getToken(fugue_sqlParser.FIELDS, 0)

        def FILEFORMAT(self):
            return self.getToken(fugue_sqlParser.FILEFORMAT, 0)

        def FIRST(self):
            return self.getToken(fugue_sqlParser.FIRST, 0)

        def FOLLOWING(self):
            return self.getToken(fugue_sqlParser.FOLLOWING, 0)

        def FOR(self):
            return self.getToken(fugue_sqlParser.FOR, 0)

        def FOREIGN(self):
            return self.getToken(fugue_sqlParser.FOREIGN, 0)

        def FORMAT(self):
            return self.getToken(fugue_sqlParser.FORMAT, 0)

        def FORMATTED(self):
            return self.getToken(fugue_sqlParser.FORMATTED, 0)

        def FROM(self):
            return self.getToken(fugue_sqlParser.FROM, 0)

        def FUNCTION(self):
            return self.getToken(fugue_sqlParser.FUNCTION, 0)

        def FUNCTIONS(self):
            return self.getToken(fugue_sqlParser.FUNCTIONS, 0)

        def GLOBAL(self):
            return self.getToken(fugue_sqlParser.GLOBAL, 0)

        def GRANT(self):
            return self.getToken(fugue_sqlParser.GRANT, 0)

        def GROUP(self):
            return self.getToken(fugue_sqlParser.GROUP, 0)

        def GROUPING(self):
            return self.getToken(fugue_sqlParser.GROUPING, 0)

        def HAVING(self):
            return self.getToken(fugue_sqlParser.HAVING, 0)

        def HOUR(self):
            return self.getToken(fugue_sqlParser.HOUR, 0)

        def IF(self):
            return self.getToken(fugue_sqlParser.IF, 0)

        def IGNORE(self):
            return self.getToken(fugue_sqlParser.IGNORE, 0)

        def IMPORT(self):
            return self.getToken(fugue_sqlParser.IMPORT, 0)

        def IN(self):
            return self.getToken(fugue_sqlParser.IN, 0)

        def INDEX(self):
            return self.getToken(fugue_sqlParser.INDEX, 0)

        def INDEXES(self):
            return self.getToken(fugue_sqlParser.INDEXES, 0)

        def INPATH(self):
            return self.getToken(fugue_sqlParser.INPATH, 0)

        def INPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)

        def INSERT(self):
            return self.getToken(fugue_sqlParser.INSERT, 0)

        def INTERVAL(self):
            return self.getToken(fugue_sqlParser.INTERVAL, 0)

        def INTO(self):
            return self.getToken(fugue_sqlParser.INTO, 0)

        def IS(self):
            return self.getToken(fugue_sqlParser.IS, 0)

        def ITEMS(self):
            return self.getToken(fugue_sqlParser.ITEMS, 0)

        def KEYS(self):
            return self.getToken(fugue_sqlParser.KEYS, 0)

        def LAST(self):
            return self.getToken(fugue_sqlParser.LAST, 0)

        def LATERAL(self):
            return self.getToken(fugue_sqlParser.LATERAL, 0)

        def LAZY(self):
            return self.getToken(fugue_sqlParser.LAZY, 0)

        def LEADING(self):
            return self.getToken(fugue_sqlParser.LEADING, 0)

        def LIKE(self):
            return self.getToken(fugue_sqlParser.LIKE, 0)

        def LIMIT(self):
            return self.getToken(fugue_sqlParser.LIMIT, 0)

        def LINES(self):
            return self.getToken(fugue_sqlParser.LINES, 0)

        def LIST(self):
            return self.getToken(fugue_sqlParser.LIST, 0)

        def LOAD(self):
            return self.getToken(fugue_sqlParser.LOAD, 0)

        def LOCAL(self):
            return self.getToken(fugue_sqlParser.LOCAL, 0)

        def LOCATION(self):
            return self.getToken(fugue_sqlParser.LOCATION, 0)

        def LOCK(self):
            return self.getToken(fugue_sqlParser.LOCK, 0)

        def LOCKS(self):
            return self.getToken(fugue_sqlParser.LOCKS, 0)

        def LOGICAL(self):
            return self.getToken(fugue_sqlParser.LOGICAL, 0)

        def MACRO(self):
            return self.getToken(fugue_sqlParser.MACRO, 0)

        def MAP(self):
            return self.getToken(fugue_sqlParser.MAP, 0)

        def MATCHED(self):
            return self.getToken(fugue_sqlParser.MATCHED, 0)

        def MERGE(self):
            return self.getToken(fugue_sqlParser.MERGE, 0)

        def MINUTE(self):
            return self.getToken(fugue_sqlParser.MINUTE, 0)

        def MONTH(self):
            return self.getToken(fugue_sqlParser.MONTH, 0)

        def MSCK(self):
            return self.getToken(fugue_sqlParser.MSCK, 0)

        def NAMESPACE(self):
            return self.getToken(fugue_sqlParser.NAMESPACE, 0)

        def NAMESPACES(self):
            return self.getToken(fugue_sqlParser.NAMESPACES, 0)

        def NO(self):
            return self.getToken(fugue_sqlParser.NO, 0)

        def NOT(self):
            return self.getToken(fugue_sqlParser.NOT, 0)

        def THENULL(self):
            return self.getToken(fugue_sqlParser.THENULL, 0)

        def THENULLS(self):
            return self.getToken(fugue_sqlParser.THENULLS, 0)

        def OF(self):
            return self.getToken(fugue_sqlParser.OF, 0)

        def ONLY(self):
            return self.getToken(fugue_sqlParser.ONLY, 0)

        def OPTION(self):
            return self.getToken(fugue_sqlParser.OPTION, 0)

        def OPTIONS(self):
            return self.getToken(fugue_sqlParser.OPTIONS, 0)

        def OR(self):
            return self.getToken(fugue_sqlParser.OR, 0)

        def ORDER(self):
            return self.getToken(fugue_sqlParser.ORDER, 0)

        def OUT(self):
            return self.getToken(fugue_sqlParser.OUT, 0)

        def OUTER(self):
            return self.getToken(fugue_sqlParser.OUTER, 0)

        def OUTPUTFORMAT(self):
            return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)

        def OVER(self):
            return self.getToken(fugue_sqlParser.OVER, 0)

        def OVERLAPS(self):
            return self.getToken(fugue_sqlParser.OVERLAPS, 0)

        def OVERLAY(self):
            return self.getToken(fugue_sqlParser.OVERLAY, 0)

        def OVERWRITE(self):
            return self.getToken(fugue_sqlParser.OVERWRITE, 0)

        def PARTITION(self):
            return self.getToken(fugue_sqlParser.PARTITION, 0)

        def PARTITIONED(self):
            return self.getToken(fugue_sqlParser.PARTITIONED, 0)

        def PARTITIONS(self):
            return self.getToken(fugue_sqlParser.PARTITIONS, 0)

        def PERCENTLIT(self):
            return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

        def PIVOT(self):
            return self.getToken(fugue_sqlParser.PIVOT, 0)

        def PLACING(self):
            return self.getToken(fugue_sqlParser.PLACING, 0)

        def POSITION(self):
            return self.getToken(fugue_sqlParser.POSITION, 0)

        def PRECEDING(self):
            return self.getToken(fugue_sqlParser.PRECEDING, 0)

        def PRIMARY(self):
            return self.getToken(fugue_sqlParser.PRIMARY, 0)

        def PRINCIPALS(self):
            return self.getToken(fugue_sqlParser.PRINCIPALS, 0)

        def PROPERTIES(self):
            return self.getToken(fugue_sqlParser.PROPERTIES, 0)

        def PURGE(self):
            return self.getToken(fugue_sqlParser.PURGE, 0)

        def QUERY(self):
            return self.getToken(fugue_sqlParser.QUERY, 0)

        def RANGE(self):
            return self.getToken(fugue_sqlParser.RANGE, 0)

        def RECORDREADER(self):
            return self.getToken(fugue_sqlParser.RECORDREADER, 0)

        def RECORDWRITER(self):
            return self.getToken(fugue_sqlParser.RECORDWRITER, 0)

        def RECOVER(self):
            return self.getToken(fugue_sqlParser.RECOVER, 0)

        def REDUCE(self):
            return self.getToken(fugue_sqlParser.REDUCE, 0)

        def REFERENCES(self):
            return self.getToken(fugue_sqlParser.REFERENCES, 0)

        def REFRESH(self):
            return self.getToken(fugue_sqlParser.REFRESH, 0)

        def RENAME(self):
            return self.getToken(fugue_sqlParser.RENAME, 0)

        def REPAIR(self):
            return self.getToken(fugue_sqlParser.REPAIR, 0)

        def REPLACE(self):
            return self.getToken(fugue_sqlParser.REPLACE, 0)

        def RESET(self):
            return self.getToken(fugue_sqlParser.RESET, 0)

        def RESTRICT(self):
            return self.getToken(fugue_sqlParser.RESTRICT, 0)

        def REVOKE(self):
            return self.getToken(fugue_sqlParser.REVOKE, 0)

        def RLIKE(self):
            return self.getToken(fugue_sqlParser.RLIKE, 0)

        def ROLE(self):
            return self.getToken(fugue_sqlParser.ROLE, 0)

        def ROLES(self):
            return self.getToken(fugue_sqlParser.ROLES, 0)

        def ROLLBACK(self):
            return self.getToken(fugue_sqlParser.ROLLBACK, 0)

        def ROLLUP(self):
            return self.getToken(fugue_sqlParser.ROLLUP, 0)

        def ROW(self):
            return self.getToken(fugue_sqlParser.ROW, 0)

        def ROWS(self):
            return self.getToken(fugue_sqlParser.ROWS, 0)

        def SCHEMA(self):
            return self.getToken(fugue_sqlParser.SCHEMA, 0)

        def SECOND(self):
            return self.getToken(fugue_sqlParser.SECOND, 0)

        def SELECT(self):
            return self.getToken(fugue_sqlParser.SELECT, 0)

        def SEPARATED(self):
            return self.getToken(fugue_sqlParser.SEPARATED, 0)

        def SERDE(self):
            return self.getToken(fugue_sqlParser.SERDE, 0)

        def SERDEPROPERTIES(self):
            return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

        def SESSION_USER(self):
            return self.getToken(fugue_sqlParser.SESSION_USER, 0)

        def SET(self):
            return self.getToken(fugue_sqlParser.SET, 0)

        def SETS(self):
            return self.getToken(fugue_sqlParser.SETS, 0)

        def SHOW(self):
            return self.getToken(fugue_sqlParser.SHOW, 0)

        def SKEWED(self):
            return self.getToken(fugue_sqlParser.SKEWED, 0)

        def SOME(self):
            return self.getToken(fugue_sqlParser.SOME, 0)

        def SORT(self):
            return self.getToken(fugue_sqlParser.SORT, 0)

        def SORTED(self):
            return self.getToken(fugue_sqlParser.SORTED, 0)

        def START(self):
            return self.getToken(fugue_sqlParser.START, 0)

        def STATISTICS(self):
            return self.getToken(fugue_sqlParser.STATISTICS, 0)

        def STORED(self):
            return self.getToken(fugue_sqlParser.STORED, 0)

        def STRATIFY(self):
            return self.getToken(fugue_sqlParser.STRATIFY, 0)

        def STRUCT(self):
            return self.getToken(fugue_sqlParser.STRUCT, 0)

        def SUBSTR(self):
            return self.getToken(fugue_sqlParser.SUBSTR, 0)

        def SUBSTRING(self):
            return self.getToken(fugue_sqlParser.SUBSTRING, 0)

        def TABLE(self):
            return self.getToken(fugue_sqlParser.TABLE, 0)

        def TABLES(self):
            return self.getToken(fugue_sqlParser.TABLES, 0)

        def TABLESAMPLE(self):
            return self.getToken(fugue_sqlParser.TABLESAMPLE, 0)

        def TBLPROPERTIES(self):
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

        def TEMPORARY(self):
            return self.getToken(fugue_sqlParser.TEMPORARY, 0)

        def TERMINATED(self):
            return self.getToken(fugue_sqlParser.TERMINATED, 0)

        def THEN(self):
            return self.getToken(fugue_sqlParser.THEN, 0)

        def TO(self):
            return self.getToken(fugue_sqlParser.TO, 0)

        def TOUCH(self):
            return self.getToken(fugue_sqlParser.TOUCH, 0)

        def TRAILING(self):
            return self.getToken(fugue_sqlParser.TRAILING, 0)

        def TRANSACTION(self):
            return self.getToken(fugue_sqlParser.TRANSACTION, 0)

        def TRANSACTIONS(self):
            return self.getToken(fugue_sqlParser.TRANSACTIONS, 0)

        def TRANSFORM(self):
            return self.getToken(fugue_sqlParser.TRANSFORM, 0)

        def TRIM(self):
            return self.getToken(fugue_sqlParser.TRIM, 0)

        def TRUE(self):
            return self.getToken(fugue_sqlParser.TRUE, 0)

        def TRUNCATE(self):
            return self.getToken(fugue_sqlParser.TRUNCATE, 0)

        def TYPE(self):
            return self.getToken(fugue_sqlParser.TYPE, 0)

        def UNARCHIVE(self):
            return self.getToken(fugue_sqlParser.UNARCHIVE, 0)

        def UNBOUNDED(self):
            return self.getToken(fugue_sqlParser.UNBOUNDED, 0)

        def UNCACHE(self):
            return self.getToken(fugue_sqlParser.UNCACHE, 0)

        def UNIQUE(self):
            return self.getToken(fugue_sqlParser.UNIQUE, 0)

        def UNKNOWN(self):
            return self.getToken(fugue_sqlParser.UNKNOWN, 0)

        def UNLOCK(self):
            return self.getToken(fugue_sqlParser.UNLOCK, 0)

        def UNSET(self):
            return self.getToken(fugue_sqlParser.UNSET, 0)

        def UPDATE(self):
            return self.getToken(fugue_sqlParser.UPDATE, 0)

        def USE(self):
            return self.getToken(fugue_sqlParser.USE, 0)

        def USER(self):
            return self.getToken(fugue_sqlParser.USER, 0)

        def VALUES(self):
            return self.getToken(fugue_sqlParser.VALUES, 0)

        def VIEW(self):
            return self.getToken(fugue_sqlParser.VIEW, 0)

        def VIEWS(self):
            return self.getToken(fugue_sqlParser.VIEWS, 0)

        def WHEN(self):
            return self.getToken(fugue_sqlParser.WHEN, 0)

        def WHERE(self):
            return self.getToken(fugue_sqlParser.WHERE, 0)

        def WINDOW(self):
            return self.getToken(fugue_sqlParser.WINDOW, 0)

        def WITH(self):
            return self.getToken(fugue_sqlParser.WITH, 0)

        def YEAR(self):
            return self.getToken(fugue_sqlParser.YEAR, 0)

        def getRuleIndex(self):
            return fugue_sqlParser.RULE_nonReserved

        def accept(self, visitor: ParseTreeVisitor):
            if hasattr(visitor, 'visitNonReserved'):
                return visitor.visitNonReserved(self)
            else:
                return visitor.visitChildren(self)

    def nonReserved(self):
        localctx = fugue_sqlParser.NonReservedContext(self, self._ctx, self.state)
        self.enterRule(localctx, 428, self.RULE_nonReserved)
        self._la = 0
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 3869
            _la = self._input.LA(1)
            if not (_la - 58 & ~63 == 0 and 1 << _la - 58 & -4398046511169 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -36610438703611937 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -72057594038061057 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 1150660908700138999 != 0) or (_la == 324)):
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
        self._predicates[48] = self.fuguePartitionNum_sempred
        self._predicates[116] = self.queryTerm_sempred
        self._predicates[171] = self.booleanExpression_sempred
        self._predicates[173] = self.valueExpression_sempred
        self._predicates[174] = self.primaryExpression_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception('No predicate with index:' + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def fuguePartitionNum_sempred(self, localctx: FuguePartitionNumContext, predIndex: int):
        if predIndex == 0:
            return self.precpred(self._ctx, 1)

    def queryTerm_sempred(self, localctx: QueryTermContext, predIndex: int):
        if predIndex == 1:
            return self.precpred(self._ctx, 3)
        if predIndex == 2:
            return self.precpred(self._ctx, 2)
        if predIndex == 3:
            return self.precpred(self._ctx, 1)

    def booleanExpression_sempred(self, localctx: BooleanExpressionContext, predIndex: int):
        if predIndex == 4:
            return self.precpred(self._ctx, 2)
        if predIndex == 5:
            return self.precpred(self._ctx, 1)

    def valueExpression_sempred(self, localctx: ValueExpressionContext, predIndex: int):
        if predIndex == 6:
            return self.precpred(self._ctx, 6)
        if predIndex == 7:
            return self.precpred(self._ctx, 5)
        if predIndex == 8:
            return self.precpred(self._ctx, 4)
        if predIndex == 9:
            return self.precpred(self._ctx, 3)
        if predIndex == 10:
            return self.precpred(self._ctx, 2)
        if predIndex == 11:
            return self.precpred(self._ctx, 1)

    def primaryExpression_sempred(self, localctx: PrimaryExpressionContext, predIndex: int):
        if predIndex == 12:
            return self.precpred(self._ctx, 8)
        if predIndex == 13:
            return self.precpred(self._ctx, 6)