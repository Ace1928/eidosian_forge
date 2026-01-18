from antlr4 import *
class fugue_sqlVisitor(ParseTreeVisitor):

    def visitFugueLanguage(self, ctx: fugue_sqlParser.FugueLanguageContext):
        return self.visitChildren(ctx)

    def visitFugueSingleStatement(self, ctx: fugue_sqlParser.FugueSingleStatementContext):
        return self.visitChildren(ctx)

    def visitFugueSingleTask(self, ctx: fugue_sqlParser.FugueSingleTaskContext):
        return self.visitChildren(ctx)

    def visitFugueNestableTask(self, ctx: fugue_sqlParser.FugueNestableTaskContext):
        return self.visitChildren(ctx)

    def visitFugueNestableTaskCollectionNoSelect(self, ctx: fugue_sqlParser.FugueNestableTaskCollectionNoSelectContext):
        return self.visitChildren(ctx)

    def visitFugueTransformTask(self, ctx: fugue_sqlParser.FugueTransformTaskContext):
        return self.visitChildren(ctx)

    def visitFugueProcessTask(self, ctx: fugue_sqlParser.FugueProcessTaskContext):
        return self.visitChildren(ctx)

    def visitFugueSaveAndUseTask(self, ctx: fugue_sqlParser.FugueSaveAndUseTaskContext):
        return self.visitChildren(ctx)

    def visitFugueRenameColumnsTask(self, ctx: fugue_sqlParser.FugueRenameColumnsTaskContext):
        return self.visitChildren(ctx)

    def visitFugueAlterColumnsTask(self, ctx: fugue_sqlParser.FugueAlterColumnsTaskContext):
        return self.visitChildren(ctx)

    def visitFugueDropColumnsTask(self, ctx: fugue_sqlParser.FugueDropColumnsTaskContext):
        return self.visitChildren(ctx)

    def visitFugueDropnaTask(self, ctx: fugue_sqlParser.FugueDropnaTaskContext):
        return self.visitChildren(ctx)

    def visitFugueFillnaTask(self, ctx: fugue_sqlParser.FugueFillnaTaskContext):
        return self.visitChildren(ctx)

    def visitFugueSampleTask(self, ctx: fugue_sqlParser.FugueSampleTaskContext):
        return self.visitChildren(ctx)

    def visitFugueTakeTask(self, ctx: fugue_sqlParser.FugueTakeTaskContext):
        return self.visitChildren(ctx)

    def visitFugueZipTask(self, ctx: fugue_sqlParser.FugueZipTaskContext):
        return self.visitChildren(ctx)

    def visitFugueCreateTask(self, ctx: fugue_sqlParser.FugueCreateTaskContext):
        return self.visitChildren(ctx)

    def visitFugueCreateDataTask(self, ctx: fugue_sqlParser.FugueCreateDataTaskContext):
        return self.visitChildren(ctx)

    def visitFugueLoadTask(self, ctx: fugue_sqlParser.FugueLoadTaskContext):
        return self.visitChildren(ctx)

    def visitFugueOutputTask(self, ctx: fugue_sqlParser.FugueOutputTaskContext):
        return self.visitChildren(ctx)

    def visitFuguePrintTask(self, ctx: fugue_sqlParser.FuguePrintTaskContext):
        return self.visitChildren(ctx)

    def visitFugueSaveTask(self, ctx: fugue_sqlParser.FugueSaveTaskContext):
        return self.visitChildren(ctx)

    def visitFugueOutputTransformTask(self, ctx: fugue_sqlParser.FugueOutputTransformTaskContext):
        return self.visitChildren(ctx)

    def visitFugueModuleTask(self, ctx: fugue_sqlParser.FugueModuleTaskContext):
        return self.visitChildren(ctx)

    def visitFugueSqlEngine(self, ctx: fugue_sqlParser.FugueSqlEngineContext):
        return self.visitChildren(ctx)

    def visitFugueSingleFile(self, ctx: fugue_sqlParser.FugueSingleFileContext):
        return self.visitChildren(ctx)

    def visitFugueLoadColumns(self, ctx: fugue_sqlParser.FugueLoadColumnsContext):
        return self.visitChildren(ctx)

    def visitFugueSaveMode(self, ctx: fugue_sqlParser.FugueSaveModeContext):
        return self.visitChildren(ctx)

    def visitFugueFileFormat(self, ctx: fugue_sqlParser.FugueFileFormatContext):
        return self.visitChildren(ctx)

    def visitFuguePath(self, ctx: fugue_sqlParser.FuguePathContext):
        return self.visitChildren(ctx)

    def visitFuguePaths(self, ctx: fugue_sqlParser.FuguePathsContext):
        return self.visitChildren(ctx)

    def visitFugueCheckpointWeak(self, ctx: fugue_sqlParser.FugueCheckpointWeakContext):
        return self.visitChildren(ctx)

    def visitFugueCheckpointStrong(self, ctx: fugue_sqlParser.FugueCheckpointStrongContext):
        return self.visitChildren(ctx)

    def visitFugueCheckpointDeterministic(self, ctx: fugue_sqlParser.FugueCheckpointDeterministicContext):
        return self.visitChildren(ctx)

    def visitFugueCheckpointNamespace(self, ctx: fugue_sqlParser.FugueCheckpointNamespaceContext):
        return self.visitChildren(ctx)

    def visitFugueYield(self, ctx: fugue_sqlParser.FugueYieldContext):
        return self.visitChildren(ctx)

    def visitFugueBroadcast(self, ctx: fugue_sqlParser.FugueBroadcastContext):
        return self.visitChildren(ctx)

    def visitFugueDataFramesList(self, ctx: fugue_sqlParser.FugueDataFramesListContext):
        return self.visitChildren(ctx)

    def visitFugueDataFramesDict(self, ctx: fugue_sqlParser.FugueDataFramesDictContext):
        return self.visitChildren(ctx)

    def visitFugueDataFramePair(self, ctx: fugue_sqlParser.FugueDataFramePairContext):
        return self.visitChildren(ctx)

    def visitFugueDataFrameSource(self, ctx: fugue_sqlParser.FugueDataFrameSourceContext):
        return self.visitChildren(ctx)

    def visitFugueDataFrameNested(self, ctx: fugue_sqlParser.FugueDataFrameNestedContext):
        return self.visitChildren(ctx)

    def visitFugueDataFrameMember(self, ctx: fugue_sqlParser.FugueDataFrameMemberContext):
        return self.visitChildren(ctx)

    def visitFugueAssignment(self, ctx: fugue_sqlParser.FugueAssignmentContext):
        return self.visitChildren(ctx)

    def visitFugueAssignmentSign(self, ctx: fugue_sqlParser.FugueAssignmentSignContext):
        return self.visitChildren(ctx)

    def visitFugueSingleOutputExtensionCommonWild(self, ctx: fugue_sqlParser.FugueSingleOutputExtensionCommonWildContext):
        return self.visitChildren(ctx)

    def visitFugueSingleOutputExtensionCommon(self, ctx: fugue_sqlParser.FugueSingleOutputExtensionCommonContext):
        return self.visitChildren(ctx)

    def visitFugueExtension(self, ctx: fugue_sqlParser.FugueExtensionContext):
        return self.visitChildren(ctx)

    def visitFugueSampleMethod(self, ctx: fugue_sqlParser.FugueSampleMethodContext):
        return self.visitChildren(ctx)

    def visitFugueZipType(self, ctx: fugue_sqlParser.FugueZipTypeContext):
        return self.visitChildren(ctx)

    def visitFuguePrepartition(self, ctx: fugue_sqlParser.FuguePrepartitionContext):
        return self.visitChildren(ctx)

    def visitFuguePartitionAlgo(self, ctx: fugue_sqlParser.FuguePartitionAlgoContext):
        return self.visitChildren(ctx)

    def visitFuguePartitionNum(self, ctx: fugue_sqlParser.FuguePartitionNumContext):
        return self.visitChildren(ctx)

    def visitFuguePartitionNumber(self, ctx: fugue_sqlParser.FuguePartitionNumberContext):
        return self.visitChildren(ctx)

    def visitFugueParamsPairs(self, ctx: fugue_sqlParser.FugueParamsPairsContext):
        return self.visitChildren(ctx)

    def visitFugueParamsObj(self, ctx: fugue_sqlParser.FugueParamsObjContext):
        return self.visitChildren(ctx)

    def visitFugueCols(self, ctx: fugue_sqlParser.FugueColsContext):
        return self.visitChildren(ctx)

    def visitFugueColsSort(self, ctx: fugue_sqlParser.FugueColsSortContext):
        return self.visitChildren(ctx)

    def visitFugueColSort(self, ctx: fugue_sqlParser.FugueColSortContext):
        return self.visitChildren(ctx)

    def visitFugueColumnIdentifier(self, ctx: fugue_sqlParser.FugueColumnIdentifierContext):
        return self.visitChildren(ctx)

    def visitFugueRenameExpression(self, ctx: fugue_sqlParser.FugueRenameExpressionContext):
        return self.visitChildren(ctx)

    def visitFugueWildSchema(self, ctx: fugue_sqlParser.FugueWildSchemaContext):
        return self.visitChildren(ctx)

    def visitFugueWildSchemaPair(self, ctx: fugue_sqlParser.FugueWildSchemaPairContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaOp(self, ctx: fugue_sqlParser.FugueSchemaOpContext):
        return self.visitChildren(ctx)

    def visitFugueSchema(self, ctx: fugue_sqlParser.FugueSchemaContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaPair(self, ctx: fugue_sqlParser.FugueSchemaPairContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaKey(self, ctx: fugue_sqlParser.FugueSchemaKeyContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaSimpleType(self, ctx: fugue_sqlParser.FugueSchemaSimpleTypeContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaListType(self, ctx: fugue_sqlParser.FugueSchemaListTypeContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaStructType(self, ctx: fugue_sqlParser.FugueSchemaStructTypeContext):
        return self.visitChildren(ctx)

    def visitFugueSchemaMapType(self, ctx: fugue_sqlParser.FugueSchemaMapTypeContext):
        return self.visitChildren(ctx)

    def visitFugueRenamePair(self, ctx: fugue_sqlParser.FugueRenamePairContext):
        return self.visitChildren(ctx)

    def visitFugueJson(self, ctx: fugue_sqlParser.FugueJsonContext):
        return self.visitChildren(ctx)

    def visitFugueJsonObj(self, ctx: fugue_sqlParser.FugueJsonObjContext):
        return self.visitChildren(ctx)

    def visitFugueJsonPairs(self, ctx: fugue_sqlParser.FugueJsonPairsContext):
        return self.visitChildren(ctx)

    def visitFugueJsonPair(self, ctx: fugue_sqlParser.FugueJsonPairContext):
        return self.visitChildren(ctx)

    def visitFugueJsonKey(self, ctx: fugue_sqlParser.FugueJsonKeyContext):
        return self.visitChildren(ctx)

    def visitFugueJsonArray(self, ctx: fugue_sqlParser.FugueJsonArrayContext):
        return self.visitChildren(ctx)

    def visitFugueJsonValue(self, ctx: fugue_sqlParser.FugueJsonValueContext):
        return self.visitChildren(ctx)

    def visitFugueJsonNumber(self, ctx: fugue_sqlParser.FugueJsonNumberContext):
        return self.visitChildren(ctx)

    def visitFugueJsonString(self, ctx: fugue_sqlParser.FugueJsonStringContext):
        return self.visitChildren(ctx)

    def visitFugueJsonBool(self, ctx: fugue_sqlParser.FugueJsonBoolContext):
        return self.visitChildren(ctx)

    def visitFugueJsonNull(self, ctx: fugue_sqlParser.FugueJsonNullContext):
        return self.visitChildren(ctx)

    def visitFugueIdentifier(self, ctx: fugue_sqlParser.FugueIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleStatement(self, ctx: fugue_sqlParser.SingleStatementContext):
        return self.visitChildren(ctx)

    def visitSingleExpression(self, ctx: fugue_sqlParser.SingleExpressionContext):
        return self.visitChildren(ctx)

    def visitSingleTableIdentifier(self, ctx: fugue_sqlParser.SingleTableIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleMultipartIdentifier(self, ctx: fugue_sqlParser.SingleMultipartIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleFunctionIdentifier(self, ctx: fugue_sqlParser.SingleFunctionIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleDataType(self, ctx: fugue_sqlParser.SingleDataTypeContext):
        return self.visitChildren(ctx)

    def visitSingleTableSchema(self, ctx: fugue_sqlParser.SingleTableSchemaContext):
        return self.visitChildren(ctx)

    def visitStatementDefault(self, ctx: fugue_sqlParser.StatementDefaultContext):
        return self.visitChildren(ctx)

    def visitDmlStatement(self, ctx: fugue_sqlParser.DmlStatementContext):
        return self.visitChildren(ctx)

    def visitUse(self, ctx: fugue_sqlParser.UseContext):
        return self.visitChildren(ctx)

    def visitCreateNamespace(self, ctx: fugue_sqlParser.CreateNamespaceContext):
        return self.visitChildren(ctx)

    def visitSetNamespaceProperties(self, ctx: fugue_sqlParser.SetNamespacePropertiesContext):
        return self.visitChildren(ctx)

    def visitSetNamespaceLocation(self, ctx: fugue_sqlParser.SetNamespaceLocationContext):
        return self.visitChildren(ctx)

    def visitDropNamespace(self, ctx: fugue_sqlParser.DropNamespaceContext):
        return self.visitChildren(ctx)

    def visitShowNamespaces(self, ctx: fugue_sqlParser.ShowNamespacesContext):
        return self.visitChildren(ctx)

    def visitCreateTable(self, ctx: fugue_sqlParser.CreateTableContext):
        return self.visitChildren(ctx)

    def visitCreateHiveTable(self, ctx: fugue_sqlParser.CreateHiveTableContext):
        return self.visitChildren(ctx)

    def visitCreateTableLike(self, ctx: fugue_sqlParser.CreateTableLikeContext):
        return self.visitChildren(ctx)

    def visitReplaceTable(self, ctx: fugue_sqlParser.ReplaceTableContext):
        return self.visitChildren(ctx)

    def visitAnalyze(self, ctx: fugue_sqlParser.AnalyzeContext):
        return self.visitChildren(ctx)

    def visitAddTableColumns(self, ctx: fugue_sqlParser.AddTableColumnsContext):
        return self.visitChildren(ctx)

    def visitRenameTableColumn(self, ctx: fugue_sqlParser.RenameTableColumnContext):
        return self.visitChildren(ctx)

    def visitDropTableColumns(self, ctx: fugue_sqlParser.DropTableColumnsContext):
        return self.visitChildren(ctx)

    def visitRenameTable(self, ctx: fugue_sqlParser.RenameTableContext):
        return self.visitChildren(ctx)

    def visitSetTableProperties(self, ctx: fugue_sqlParser.SetTablePropertiesContext):
        return self.visitChildren(ctx)

    def visitUnsetTableProperties(self, ctx: fugue_sqlParser.UnsetTablePropertiesContext):
        return self.visitChildren(ctx)

    def visitAlterTableAlterColumn(self, ctx: fugue_sqlParser.AlterTableAlterColumnContext):
        return self.visitChildren(ctx)

    def visitHiveChangeColumn(self, ctx: fugue_sqlParser.HiveChangeColumnContext):
        return self.visitChildren(ctx)

    def visitHiveReplaceColumns(self, ctx: fugue_sqlParser.HiveReplaceColumnsContext):
        return self.visitChildren(ctx)

    def visitSetTableSerDe(self, ctx: fugue_sqlParser.SetTableSerDeContext):
        return self.visitChildren(ctx)

    def visitAddTablePartition(self, ctx: fugue_sqlParser.AddTablePartitionContext):
        return self.visitChildren(ctx)

    def visitRenameTablePartition(self, ctx: fugue_sqlParser.RenameTablePartitionContext):
        return self.visitChildren(ctx)

    def visitDropTablePartitions(self, ctx: fugue_sqlParser.DropTablePartitionsContext):
        return self.visitChildren(ctx)

    def visitSetTableLocation(self, ctx: fugue_sqlParser.SetTableLocationContext):
        return self.visitChildren(ctx)

    def visitRecoverPartitions(self, ctx: fugue_sqlParser.RecoverPartitionsContext):
        return self.visitChildren(ctx)

    def visitDropTable(self, ctx: fugue_sqlParser.DropTableContext):
        return self.visitChildren(ctx)

    def visitDropView(self, ctx: fugue_sqlParser.DropViewContext):
        return self.visitChildren(ctx)

    def visitCreateView(self, ctx: fugue_sqlParser.CreateViewContext):
        return self.visitChildren(ctx)

    def visitCreateTempViewUsing(self, ctx: fugue_sqlParser.CreateTempViewUsingContext):
        return self.visitChildren(ctx)

    def visitAlterViewQuery(self, ctx: fugue_sqlParser.AlterViewQueryContext):
        return self.visitChildren(ctx)

    def visitCreateFunction(self, ctx: fugue_sqlParser.CreateFunctionContext):
        return self.visitChildren(ctx)

    def visitDropFunction(self, ctx: fugue_sqlParser.DropFunctionContext):
        return self.visitChildren(ctx)

    def visitExplain(self, ctx: fugue_sqlParser.ExplainContext):
        return self.visitChildren(ctx)

    def visitShowTables(self, ctx: fugue_sqlParser.ShowTablesContext):
        return self.visitChildren(ctx)

    def visitShowTable(self, ctx: fugue_sqlParser.ShowTableContext):
        return self.visitChildren(ctx)

    def visitShowTblProperties(self, ctx: fugue_sqlParser.ShowTblPropertiesContext):
        return self.visitChildren(ctx)

    def visitShowColumns(self, ctx: fugue_sqlParser.ShowColumnsContext):
        return self.visitChildren(ctx)

    def visitShowViews(self, ctx: fugue_sqlParser.ShowViewsContext):
        return self.visitChildren(ctx)

    def visitShowPartitions(self, ctx: fugue_sqlParser.ShowPartitionsContext):
        return self.visitChildren(ctx)

    def visitShowFunctions(self, ctx: fugue_sqlParser.ShowFunctionsContext):
        return self.visitChildren(ctx)

    def visitShowCreateTable(self, ctx: fugue_sqlParser.ShowCreateTableContext):
        return self.visitChildren(ctx)

    def visitShowCurrentNamespace(self, ctx: fugue_sqlParser.ShowCurrentNamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeFunction(self, ctx: fugue_sqlParser.DescribeFunctionContext):
        return self.visitChildren(ctx)

    def visitDescribeNamespace(self, ctx: fugue_sqlParser.DescribeNamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeRelation(self, ctx: fugue_sqlParser.DescribeRelationContext):
        return self.visitChildren(ctx)

    def visitDescribeQuery(self, ctx: fugue_sqlParser.DescribeQueryContext):
        return self.visitChildren(ctx)

    def visitCommentNamespace(self, ctx: fugue_sqlParser.CommentNamespaceContext):
        return self.visitChildren(ctx)

    def visitCommentTable(self, ctx: fugue_sqlParser.CommentTableContext):
        return self.visitChildren(ctx)

    def visitRefreshTable(self, ctx: fugue_sqlParser.RefreshTableContext):
        return self.visitChildren(ctx)

    def visitRefreshResource(self, ctx: fugue_sqlParser.RefreshResourceContext):
        return self.visitChildren(ctx)

    def visitCacheTable(self, ctx: fugue_sqlParser.CacheTableContext):
        return self.visitChildren(ctx)

    def visitUncacheTable(self, ctx: fugue_sqlParser.UncacheTableContext):
        return self.visitChildren(ctx)

    def visitClearCache(self, ctx: fugue_sqlParser.ClearCacheContext):
        return self.visitChildren(ctx)

    def visitLoadData(self, ctx: fugue_sqlParser.LoadDataContext):
        return self.visitChildren(ctx)

    def visitTruncateTable(self, ctx: fugue_sqlParser.TruncateTableContext):
        return self.visitChildren(ctx)

    def visitRepairTable(self, ctx: fugue_sqlParser.RepairTableContext):
        return self.visitChildren(ctx)

    def visitManageResource(self, ctx: fugue_sqlParser.ManageResourceContext):
        return self.visitChildren(ctx)

    def visitFailNativeCommand(self, ctx: fugue_sqlParser.FailNativeCommandContext):
        return self.visitChildren(ctx)

    def visitSetConfiguration(self, ctx: fugue_sqlParser.SetConfigurationContext):
        return self.visitChildren(ctx)

    def visitResetConfiguration(self, ctx: fugue_sqlParser.ResetConfigurationContext):
        return self.visitChildren(ctx)

    def visitUnsupportedHiveNativeCommands(self, ctx: fugue_sqlParser.UnsupportedHiveNativeCommandsContext):
        return self.visitChildren(ctx)

    def visitCreateTableHeader(self, ctx: fugue_sqlParser.CreateTableHeaderContext):
        return self.visitChildren(ctx)

    def visitReplaceTableHeader(self, ctx: fugue_sqlParser.ReplaceTableHeaderContext):
        return self.visitChildren(ctx)

    def visitBucketSpec(self, ctx: fugue_sqlParser.BucketSpecContext):
        return self.visitChildren(ctx)

    def visitSkewSpec(self, ctx: fugue_sqlParser.SkewSpecContext):
        return self.visitChildren(ctx)

    def visitLocationSpec(self, ctx: fugue_sqlParser.LocationSpecContext):
        return self.visitChildren(ctx)

    def visitCommentSpec(self, ctx: fugue_sqlParser.CommentSpecContext):
        return self.visitChildren(ctx)

    def visitQuery(self, ctx: fugue_sqlParser.QueryContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteTable(self, ctx: fugue_sqlParser.InsertOverwriteTableContext):
        return self.visitChildren(ctx)

    def visitInsertIntoTable(self, ctx: fugue_sqlParser.InsertIntoTableContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteHiveDir(self, ctx: fugue_sqlParser.InsertOverwriteHiveDirContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteDir(self, ctx: fugue_sqlParser.InsertOverwriteDirContext):
        return self.visitChildren(ctx)

    def visitPartitionSpecLocation(self, ctx: fugue_sqlParser.PartitionSpecLocationContext):
        return self.visitChildren(ctx)

    def visitPartitionSpec(self, ctx: fugue_sqlParser.PartitionSpecContext):
        return self.visitChildren(ctx)

    def visitPartitionVal(self, ctx: fugue_sqlParser.PartitionValContext):
        return self.visitChildren(ctx)

    def visitTheNamespace(self, ctx: fugue_sqlParser.TheNamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeFuncName(self, ctx: fugue_sqlParser.DescribeFuncNameContext):
        return self.visitChildren(ctx)

    def visitDescribeColName(self, ctx: fugue_sqlParser.DescribeColNameContext):
        return self.visitChildren(ctx)

    def visitCtes(self, ctx: fugue_sqlParser.CtesContext):
        return self.visitChildren(ctx)

    def visitNamedQuery(self, ctx: fugue_sqlParser.NamedQueryContext):
        return self.visitChildren(ctx)

    def visitTableProvider(self, ctx: fugue_sqlParser.TableProviderContext):
        return self.visitChildren(ctx)

    def visitCreateTableClauses(self, ctx: fugue_sqlParser.CreateTableClausesContext):
        return self.visitChildren(ctx)

    def visitTablePropertyList(self, ctx: fugue_sqlParser.TablePropertyListContext):
        return self.visitChildren(ctx)

    def visitTableProperty(self, ctx: fugue_sqlParser.TablePropertyContext):
        return self.visitChildren(ctx)

    def visitTablePropertyKey(self, ctx: fugue_sqlParser.TablePropertyKeyContext):
        return self.visitChildren(ctx)

    def visitTablePropertyValue(self, ctx: fugue_sqlParser.TablePropertyValueContext):
        return self.visitChildren(ctx)

    def visitConstantList(self, ctx: fugue_sqlParser.ConstantListContext):
        return self.visitChildren(ctx)

    def visitNestedConstantList(self, ctx: fugue_sqlParser.NestedConstantListContext):
        return self.visitChildren(ctx)

    def visitCreateFileFormat(self, ctx: fugue_sqlParser.CreateFileFormatContext):
        return self.visitChildren(ctx)

    def visitTableFileFormat(self, ctx: fugue_sqlParser.TableFileFormatContext):
        return self.visitChildren(ctx)

    def visitGenericFileFormat(self, ctx: fugue_sqlParser.GenericFileFormatContext):
        return self.visitChildren(ctx)

    def visitStorageHandler(self, ctx: fugue_sqlParser.StorageHandlerContext):
        return self.visitChildren(ctx)

    def visitResource(self, ctx: fugue_sqlParser.ResourceContext):
        return self.visitChildren(ctx)

    def visitSingleInsertQuery(self, ctx: fugue_sqlParser.SingleInsertQueryContext):
        return self.visitChildren(ctx)

    def visitMultiInsertQuery(self, ctx: fugue_sqlParser.MultiInsertQueryContext):
        return self.visitChildren(ctx)

    def visitDeleteFromTable(self, ctx: fugue_sqlParser.DeleteFromTableContext):
        return self.visitChildren(ctx)

    def visitUpdateTable(self, ctx: fugue_sqlParser.UpdateTableContext):
        return self.visitChildren(ctx)

    def visitMergeIntoTable(self, ctx: fugue_sqlParser.MergeIntoTableContext):
        return self.visitChildren(ctx)

    def visitQueryOrganization(self, ctx: fugue_sqlParser.QueryOrganizationContext):
        return self.visitChildren(ctx)

    def visitMultiInsertQueryBody(self, ctx: fugue_sqlParser.MultiInsertQueryBodyContext):
        return self.visitChildren(ctx)

    def visitQueryTermDefault(self, ctx: fugue_sqlParser.QueryTermDefaultContext):
        return self.visitChildren(ctx)

    def visitFugueTerm(self, ctx: fugue_sqlParser.FugueTermContext):
        return self.visitChildren(ctx)

    def visitSetOperation(self, ctx: fugue_sqlParser.SetOperationContext):
        return self.visitChildren(ctx)

    def visitQueryPrimaryDefault(self, ctx: fugue_sqlParser.QueryPrimaryDefaultContext):
        return self.visitChildren(ctx)

    def visitFromStmt(self, ctx: fugue_sqlParser.FromStmtContext):
        return self.visitChildren(ctx)

    def visitTable(self, ctx: fugue_sqlParser.TableContext):
        return self.visitChildren(ctx)

    def visitInlineTableDefault1(self, ctx: fugue_sqlParser.InlineTableDefault1Context):
        return self.visitChildren(ctx)

    def visitSortItem(self, ctx: fugue_sqlParser.SortItemContext):
        return self.visitChildren(ctx)

    def visitFromStatement(self, ctx: fugue_sqlParser.FromStatementContext):
        return self.visitChildren(ctx)

    def visitFromStatementBody(self, ctx: fugue_sqlParser.FromStatementBodyContext):
        return self.visitChildren(ctx)

    def visitTransformQuerySpecification(self, ctx: fugue_sqlParser.TransformQuerySpecificationContext):
        return self.visitChildren(ctx)

    def visitRegularQuerySpecification(self, ctx: fugue_sqlParser.RegularQuerySpecificationContext):
        return self.visitChildren(ctx)

    def visitOptionalFromClause(self, ctx: fugue_sqlParser.OptionalFromClauseContext):
        return self.visitChildren(ctx)

    def visitTransformClause(self, ctx: fugue_sqlParser.TransformClauseContext):
        return self.visitChildren(ctx)

    def visitSelectClause(self, ctx: fugue_sqlParser.SelectClauseContext):
        return self.visitChildren(ctx)

    def visitSetClause(self, ctx: fugue_sqlParser.SetClauseContext):
        return self.visitChildren(ctx)

    def visitMatchedClause(self, ctx: fugue_sqlParser.MatchedClauseContext):
        return self.visitChildren(ctx)

    def visitNotMatchedClause(self, ctx: fugue_sqlParser.NotMatchedClauseContext):
        return self.visitChildren(ctx)

    def visitMatchedAction(self, ctx: fugue_sqlParser.MatchedActionContext):
        return self.visitChildren(ctx)

    def visitNotMatchedAction(self, ctx: fugue_sqlParser.NotMatchedActionContext):
        return self.visitChildren(ctx)

    def visitAssignmentList(self, ctx: fugue_sqlParser.AssignmentListContext):
        return self.visitChildren(ctx)

    def visitAssignment(self, ctx: fugue_sqlParser.AssignmentContext):
        return self.visitChildren(ctx)

    def visitWhereClause(self, ctx: fugue_sqlParser.WhereClauseContext):
        return self.visitChildren(ctx)

    def visitHavingClause(self, ctx: fugue_sqlParser.HavingClauseContext):
        return self.visitChildren(ctx)

    def visitHint(self, ctx: fugue_sqlParser.HintContext):
        return self.visitChildren(ctx)

    def visitHintStatement(self, ctx: fugue_sqlParser.HintStatementContext):
        return self.visitChildren(ctx)

    def visitFromClause(self, ctx: fugue_sqlParser.FromClauseContext):
        return self.visitChildren(ctx)

    def visitAggregationClause(self, ctx: fugue_sqlParser.AggregationClauseContext):
        return self.visitChildren(ctx)

    def visitGroupingSet(self, ctx: fugue_sqlParser.GroupingSetContext):
        return self.visitChildren(ctx)

    def visitPivotClause(self, ctx: fugue_sqlParser.PivotClauseContext):
        return self.visitChildren(ctx)

    def visitPivotColumn(self, ctx: fugue_sqlParser.PivotColumnContext):
        return self.visitChildren(ctx)

    def visitPivotValue(self, ctx: fugue_sqlParser.PivotValueContext):
        return self.visitChildren(ctx)

    def visitLateralView(self, ctx: fugue_sqlParser.LateralViewContext):
        return self.visitChildren(ctx)

    def visitSetQuantifier(self, ctx: fugue_sqlParser.SetQuantifierContext):
        return self.visitChildren(ctx)

    def visitRelation(self, ctx: fugue_sqlParser.RelationContext):
        return self.visitChildren(ctx)

    def visitJoinRelation(self, ctx: fugue_sqlParser.JoinRelationContext):
        return self.visitChildren(ctx)

    def visitJoinType(self, ctx: fugue_sqlParser.JoinTypeContext):
        return self.visitChildren(ctx)

    def visitJoinCriteria(self, ctx: fugue_sqlParser.JoinCriteriaContext):
        return self.visitChildren(ctx)

    def visitSample(self, ctx: fugue_sqlParser.SampleContext):
        return self.visitChildren(ctx)

    def visitSampleByPercentile(self, ctx: fugue_sqlParser.SampleByPercentileContext):
        return self.visitChildren(ctx)

    def visitSampleByRows(self, ctx: fugue_sqlParser.SampleByRowsContext):
        return self.visitChildren(ctx)

    def visitSampleByBucket(self, ctx: fugue_sqlParser.SampleByBucketContext):
        return self.visitChildren(ctx)

    def visitSampleByBytes(self, ctx: fugue_sqlParser.SampleByBytesContext):
        return self.visitChildren(ctx)

    def visitIdentifierList(self, ctx: fugue_sqlParser.IdentifierListContext):
        return self.visitChildren(ctx)

    def visitIdentifierSeq(self, ctx: fugue_sqlParser.IdentifierSeqContext):
        return self.visitChildren(ctx)

    def visitOrderedIdentifierList(self, ctx: fugue_sqlParser.OrderedIdentifierListContext):
        return self.visitChildren(ctx)

    def visitOrderedIdentifier(self, ctx: fugue_sqlParser.OrderedIdentifierContext):
        return self.visitChildren(ctx)

    def visitIdentifierCommentList(self, ctx: fugue_sqlParser.IdentifierCommentListContext):
        return self.visitChildren(ctx)

    def visitIdentifierComment(self, ctx: fugue_sqlParser.IdentifierCommentContext):
        return self.visitChildren(ctx)

    def visitTableName(self, ctx: fugue_sqlParser.TableNameContext):
        return self.visitChildren(ctx)

    def visitAliasedQuery(self, ctx: fugue_sqlParser.AliasedQueryContext):
        return self.visitChildren(ctx)

    def visitAliasedRelation(self, ctx: fugue_sqlParser.AliasedRelationContext):
        return self.visitChildren(ctx)

    def visitInlineTableDefault2(self, ctx: fugue_sqlParser.InlineTableDefault2Context):
        return self.visitChildren(ctx)

    def visitTableValuedFunction(self, ctx: fugue_sqlParser.TableValuedFunctionContext):
        return self.visitChildren(ctx)

    def visitInlineTable(self, ctx: fugue_sqlParser.InlineTableContext):
        return self.visitChildren(ctx)

    def visitFunctionTable(self, ctx: fugue_sqlParser.FunctionTableContext):
        return self.visitChildren(ctx)

    def visitTableAlias(self, ctx: fugue_sqlParser.TableAliasContext):
        return self.visitChildren(ctx)

    def visitRowFormatSerde(self, ctx: fugue_sqlParser.RowFormatSerdeContext):
        return self.visitChildren(ctx)

    def visitRowFormatDelimited(self, ctx: fugue_sqlParser.RowFormatDelimitedContext):
        return self.visitChildren(ctx)

    def visitMultipartIdentifierList(self, ctx: fugue_sqlParser.MultipartIdentifierListContext):
        return self.visitChildren(ctx)

    def visitMultipartIdentifier(self, ctx: fugue_sqlParser.MultipartIdentifierContext):
        return self.visitChildren(ctx)

    def visitTableIdentifier(self, ctx: fugue_sqlParser.TableIdentifierContext):
        return self.visitChildren(ctx)

    def visitFunctionIdentifier(self, ctx: fugue_sqlParser.FunctionIdentifierContext):
        return self.visitChildren(ctx)

    def visitNamedExpression(self, ctx: fugue_sqlParser.NamedExpressionContext):
        return self.visitChildren(ctx)

    def visitNamedExpressionSeq(self, ctx: fugue_sqlParser.NamedExpressionSeqContext):
        return self.visitChildren(ctx)

    def visitTransformList(self, ctx: fugue_sqlParser.TransformListContext):
        return self.visitChildren(ctx)

    def visitIdentityTransform(self, ctx: fugue_sqlParser.IdentityTransformContext):
        return self.visitChildren(ctx)

    def visitApplyTransform(self, ctx: fugue_sqlParser.ApplyTransformContext):
        return self.visitChildren(ctx)

    def visitTransformArgument(self, ctx: fugue_sqlParser.TransformArgumentContext):
        return self.visitChildren(ctx)

    def visitExpression(self, ctx: fugue_sqlParser.ExpressionContext):
        return self.visitChildren(ctx)

    def visitLogicalNot(self, ctx: fugue_sqlParser.LogicalNotContext):
        return self.visitChildren(ctx)

    def visitPredicated(self, ctx: fugue_sqlParser.PredicatedContext):
        return self.visitChildren(ctx)

    def visitExists(self, ctx: fugue_sqlParser.ExistsContext):
        return self.visitChildren(ctx)

    def visitLogicalBinary(self, ctx: fugue_sqlParser.LogicalBinaryContext):
        return self.visitChildren(ctx)

    def visitPredicate(self, ctx: fugue_sqlParser.PredicateContext):
        return self.visitChildren(ctx)

    def visitValueExpressionDefault(self, ctx: fugue_sqlParser.ValueExpressionDefaultContext):
        return self.visitChildren(ctx)

    def visitComparison(self, ctx: fugue_sqlParser.ComparisonContext):
        return self.visitChildren(ctx)

    def visitArithmeticBinary(self, ctx: fugue_sqlParser.ArithmeticBinaryContext):
        return self.visitChildren(ctx)

    def visitArithmeticUnary(self, ctx: fugue_sqlParser.ArithmeticUnaryContext):
        return self.visitChildren(ctx)

    def visitStruct(self, ctx: fugue_sqlParser.StructContext):
        return self.visitChildren(ctx)

    def visitDereference(self, ctx: fugue_sqlParser.DereferenceContext):
        return self.visitChildren(ctx)

    def visitSimpleCase(self, ctx: fugue_sqlParser.SimpleCaseContext):
        return self.visitChildren(ctx)

    def visitColumnReference(self, ctx: fugue_sqlParser.ColumnReferenceContext):
        return self.visitChildren(ctx)

    def visitRowConstructor(self, ctx: fugue_sqlParser.RowConstructorContext):
        return self.visitChildren(ctx)

    def visitLast(self, ctx: fugue_sqlParser.LastContext):
        return self.visitChildren(ctx)

    def visitStar(self, ctx: fugue_sqlParser.StarContext):
        return self.visitChildren(ctx)

    def visitOverlay(self, ctx: fugue_sqlParser.OverlayContext):
        return self.visitChildren(ctx)

    def visitSubscript(self, ctx: fugue_sqlParser.SubscriptContext):
        return self.visitChildren(ctx)

    def visitSubqueryExpression(self, ctx: fugue_sqlParser.SubqueryExpressionContext):
        return self.visitChildren(ctx)

    def visitSubstring(self, ctx: fugue_sqlParser.SubstringContext):
        return self.visitChildren(ctx)

    def visitCurrentDatetime(self, ctx: fugue_sqlParser.CurrentDatetimeContext):
        return self.visitChildren(ctx)

    def visitCast(self, ctx: fugue_sqlParser.CastContext):
        return self.visitChildren(ctx)

    def visitConstantDefault(self, ctx: fugue_sqlParser.ConstantDefaultContext):
        return self.visitChildren(ctx)

    def visitLambda(self, ctx: fugue_sqlParser.LambdaContext):
        return self.visitChildren(ctx)

    def visitParenthesizedExpression(self, ctx: fugue_sqlParser.ParenthesizedExpressionContext):
        return self.visitChildren(ctx)

    def visitExtract(self, ctx: fugue_sqlParser.ExtractContext):
        return self.visitChildren(ctx)

    def visitTrim(self, ctx: fugue_sqlParser.TrimContext):
        return self.visitChildren(ctx)

    def visitFunctionCall(self, ctx: fugue_sqlParser.FunctionCallContext):
        return self.visitChildren(ctx)

    def visitSearchedCase(self, ctx: fugue_sqlParser.SearchedCaseContext):
        return self.visitChildren(ctx)

    def visitPosition(self, ctx: fugue_sqlParser.PositionContext):
        return self.visitChildren(ctx)

    def visitFirst(self, ctx: fugue_sqlParser.FirstContext):
        return self.visitChildren(ctx)

    def visitNullLiteral(self, ctx: fugue_sqlParser.NullLiteralContext):
        return self.visitChildren(ctx)

    def visitIntervalLiteral(self, ctx: fugue_sqlParser.IntervalLiteralContext):
        return self.visitChildren(ctx)

    def visitTypeConstructor(self, ctx: fugue_sqlParser.TypeConstructorContext):
        return self.visitChildren(ctx)

    def visitNumericLiteral(self, ctx: fugue_sqlParser.NumericLiteralContext):
        return self.visitChildren(ctx)

    def visitBooleanLiteral(self, ctx: fugue_sqlParser.BooleanLiteralContext):
        return self.visitChildren(ctx)

    def visitStringLiteral(self, ctx: fugue_sqlParser.StringLiteralContext):
        return self.visitChildren(ctx)

    def visitComparisonOperator(self, ctx: fugue_sqlParser.ComparisonOperatorContext):
        return self.visitChildren(ctx)

    def visitComparisonEqualOperator(self, ctx: fugue_sqlParser.ComparisonEqualOperatorContext):
        return self.visitChildren(ctx)

    def visitArithmeticOperator(self, ctx: fugue_sqlParser.ArithmeticOperatorContext):
        return self.visitChildren(ctx)

    def visitPredicateOperator(self, ctx: fugue_sqlParser.PredicateOperatorContext):
        return self.visitChildren(ctx)

    def visitBooleanValue(self, ctx: fugue_sqlParser.BooleanValueContext):
        return self.visitChildren(ctx)

    def visitInterval(self, ctx: fugue_sqlParser.IntervalContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingMultiUnitsInterval(self, ctx: fugue_sqlParser.ErrorCapturingMultiUnitsIntervalContext):
        return self.visitChildren(ctx)

    def visitMultiUnitsInterval(self, ctx: fugue_sqlParser.MultiUnitsIntervalContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingUnitToUnitInterval(self, ctx: fugue_sqlParser.ErrorCapturingUnitToUnitIntervalContext):
        return self.visitChildren(ctx)

    def visitUnitToUnitInterval(self, ctx: fugue_sqlParser.UnitToUnitIntervalContext):
        return self.visitChildren(ctx)

    def visitIntervalValue(self, ctx: fugue_sqlParser.IntervalValueContext):
        return self.visitChildren(ctx)

    def visitIntervalUnit(self, ctx: fugue_sqlParser.IntervalUnitContext):
        return self.visitChildren(ctx)

    def visitColPosition(self, ctx: fugue_sqlParser.ColPositionContext):
        return self.visitChildren(ctx)

    def visitComplexDataType(self, ctx: fugue_sqlParser.ComplexDataTypeContext):
        return self.visitChildren(ctx)

    def visitPrimitiveDataType(self, ctx: fugue_sqlParser.PrimitiveDataTypeContext):
        return self.visitChildren(ctx)

    def visitQualifiedColTypeWithPositionList(self, ctx: fugue_sqlParser.QualifiedColTypeWithPositionListContext):
        return self.visitChildren(ctx)

    def visitQualifiedColTypeWithPosition(self, ctx: fugue_sqlParser.QualifiedColTypeWithPositionContext):
        return self.visitChildren(ctx)

    def visitColTypeList(self, ctx: fugue_sqlParser.ColTypeListContext):
        return self.visitChildren(ctx)

    def visitColType(self, ctx: fugue_sqlParser.ColTypeContext):
        return self.visitChildren(ctx)

    def visitComplexColTypeList(self, ctx: fugue_sqlParser.ComplexColTypeListContext):
        return self.visitChildren(ctx)

    def visitComplexColType(self, ctx: fugue_sqlParser.ComplexColTypeContext):
        return self.visitChildren(ctx)

    def visitWhenClause(self, ctx: fugue_sqlParser.WhenClauseContext):
        return self.visitChildren(ctx)

    def visitWindowClause(self, ctx: fugue_sqlParser.WindowClauseContext):
        return self.visitChildren(ctx)

    def visitNamedWindow(self, ctx: fugue_sqlParser.NamedWindowContext):
        return self.visitChildren(ctx)

    def visitWindowRef(self, ctx: fugue_sqlParser.WindowRefContext):
        return self.visitChildren(ctx)

    def visitWindowDef(self, ctx: fugue_sqlParser.WindowDefContext):
        return self.visitChildren(ctx)

    def visitWindowFrame(self, ctx: fugue_sqlParser.WindowFrameContext):
        return self.visitChildren(ctx)

    def visitFrameBound(self, ctx: fugue_sqlParser.FrameBoundContext):
        return self.visitChildren(ctx)

    def visitQualifiedNameList(self, ctx: fugue_sqlParser.QualifiedNameListContext):
        return self.visitChildren(ctx)

    def visitFunctionName(self, ctx: fugue_sqlParser.FunctionNameContext):
        return self.visitChildren(ctx)

    def visitQualifiedName(self, ctx: fugue_sqlParser.QualifiedNameContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingIdentifier(self, ctx: fugue_sqlParser.ErrorCapturingIdentifierContext):
        return self.visitChildren(ctx)

    def visitErrorIdent(self, ctx: fugue_sqlParser.ErrorIdentContext):
        return self.visitChildren(ctx)

    def visitIdentifier(self, ctx: fugue_sqlParser.IdentifierContext):
        return self.visitChildren(ctx)

    def visitUnquotedIdentifier(self, ctx: fugue_sqlParser.UnquotedIdentifierContext):
        return self.visitChildren(ctx)

    def visitQuotedIdentifierAlternative(self, ctx: fugue_sqlParser.QuotedIdentifierAlternativeContext):
        return self.visitChildren(ctx)

    def visitQuotedIdentifier(self, ctx: fugue_sqlParser.QuotedIdentifierContext):
        return self.visitChildren(ctx)

    def visitExponentLiteral(self, ctx: fugue_sqlParser.ExponentLiteralContext):
        return self.visitChildren(ctx)

    def visitDecimalLiteral(self, ctx: fugue_sqlParser.DecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitLegacyDecimalLiteral(self, ctx: fugue_sqlParser.LegacyDecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitIntegerLiteral(self, ctx: fugue_sqlParser.IntegerLiteralContext):
        return self.visitChildren(ctx)

    def visitBigIntLiteral(self, ctx: fugue_sqlParser.BigIntLiteralContext):
        return self.visitChildren(ctx)

    def visitSmallIntLiteral(self, ctx: fugue_sqlParser.SmallIntLiteralContext):
        return self.visitChildren(ctx)

    def visitTinyIntLiteral(self, ctx: fugue_sqlParser.TinyIntLiteralContext):
        return self.visitChildren(ctx)

    def visitDoubleLiteral(self, ctx: fugue_sqlParser.DoubleLiteralContext):
        return self.visitChildren(ctx)

    def visitBigDecimalLiteral(self, ctx: fugue_sqlParser.BigDecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitAlterColumnAction(self, ctx: fugue_sqlParser.AlterColumnActionContext):
        return self.visitChildren(ctx)

    def visitAnsiNonReserved(self, ctx: fugue_sqlParser.AnsiNonReservedContext):
        return self.visitChildren(ctx)

    def visitStrictNonReserved(self, ctx: fugue_sqlParser.StrictNonReservedContext):
        return self.visitChildren(ctx)

    def visitNonReserved(self, ctx: fugue_sqlParser.NonReservedContext):
        return self.visitChildren(ctx)