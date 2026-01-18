from warnings import warn
from rdkit import RDConfig
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
class CompositeRun:
    """ class to store parameters for and results from Composite building

   This class has a default set of fields which are added to the database.

   By default these fields are stored in a tuple, so they are immutable.  This
     is probably what you want.


  """
    fields = (('rundate', 'varchar(32)'), ('dbName', 'varchar(200)'), ('dbWhat', 'varchar(200)'), ('dbWhere', 'varchar(200)'), ('dbJoin', 'varchar(200)'), ('tableName', 'varchar(80)'), ('note', 'varchar(120)'), ('shuffled', 'smallint'), ('randomized', 'smallint'), ('overall_error', 'float'), ('holdout_error', 'float'), ('overall_fraction_dropped', 'float'), ('holdout_fraction_dropped', 'float'), ('overall_correct_conf', 'float'), ('overall_incorrect_conf', 'float'), ('holdout_correct_conf', 'float'), ('holdout_incorrect_conf', 'float'), ('overall_result_matrix', 'varchar(256)'), ('holdout_result_matrix', 'varchar(256)'), ('threshold', 'float'), ('splitFrac', 'float'), ('filterFrac', 'float'), ('filterVal', 'float'), ('modelFilterVal', 'float'), ('modelFilterFrac', 'float'), ('nModels', 'int'), ('limitDepth', 'int'), ('bayesModels', 'int'), ('qBoundCount', 'varchar(3000)'), ('activityBoundsVals', 'varchar(200)'), ('cmd', 'varchar(500)'), ('model', DbModule.binaryTypeName))

    def _CreateTable(self, cn, tblName):
        """ *Internal Use only*

    """
        names = map(lambda x: x.strip().upper(), cn.GetTableNames())
        if tblName.upper() not in names:
            curs = cn.GetCursor()
            fmt = []
            for name, value in self.fields:
                fmt.append('%s %s' % (name, value))
            fmtStr = ','.join(fmt)
            curs.execute('create table %s (%s)' % (tblName, fmtStr))
            cn.Commit()
        else:
            heads = [x.upper() for x in cn.GetColumnNames()]
            curs = cn.GetCursor()
            for name, value in self.fields:
                if name.upper() not in heads:
                    curs.execute('alter table %s add %s %s' % (tblName, name, value))
            cn.Commit()

    def Store(self, db='models.gdb', table='results', user='sysdba', password='masterkey'):
        """ adds the result to a database

      **Arguments**

        - db: name of the database to use

        - table: name of the table to use

        - user&password: connection information

    """
        cn = DbConnect(db, table, user, password)
        curs = cn.GetCursor()
        self._CreateTable(cn, table)
        cols = []
        vals = []
        for name, _ in self.fields:
            try:
                v = getattr(self, name)
            except AttributeError:
                pass
            else:
                cols.append('%s' % name)
                vals.append(v)
        nToDo = len(vals)
        qs = ','.join([DbModule.placeHolder] * nToDo)
        vals = tuple(vals)
        cmd = 'insert into %s (%s) values (%s)' % (table, ','.join(cols), qs)
        curs.execute(cmd, vals)
        cn.Commit()

    def GetDataSet(self, **kwargs):
        """ Returns a MLDataSet pulled from a database using our stored
    values.

    """
        from rdkit.ML.Data import DataUtils
        data = DataUtils.DBToData(self.dbName, self.tableName, user=self.dbUser, password=self.dbPassword, what=self.dbWhat, where=self.dbWhere, join=self.dbJoin, **kwargs)
        return data

    def GetDataSetInfo(self, **kwargs):
        """ Returns a MLDataSet pulled from a database using our stored
    values.

    """
        conn = DbConnect(self.dbName, self.tableName)
        res = conn.GetColumnNamesAndTypes(join=self.dbJoin, what=self.dbWhat, where=self.dbWhere)
        return res