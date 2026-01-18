from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
class LazyDBBase:

    def __init__(self, dbcache: Any, config: LazyDBConfig):
        self.config = config
        self.cache = dbcache
        self.alive = True
        self.lock = threading.RLock()
        self.setup_db_schema()
        self.init_db()
        self.migrate_db()
        self.finalize_init()

    def init_db(self):
        if self.cache.exists:
            dbdata = self.cache.restore()
            self._db = dbdata['db']
            self._alivetime = dbdata['timer']
            self.metrics = dbdata['metrics']
            self.metrics.num_loaded += 1
            self.metrics.last_load = tstamp()
        else:
            self._alivetime = timer()
            self.metrics = LazyDBSaveMetrics(created=tstamp())

    def setup_db_schema(self):
        self._db = {}
        self._base_schemas = {}
        self.__class__.__name__ = 'LazyDB' if not self.config.dbname else f'{self.config.dbname}_LazyDB'
        logger.info(f'Initializing {self.__class__.__name__}')
        logger.info(f'[{self.dbname} Setup]: Setting Up DB Schema')
        if self.config.autouser:
            logger.info(f'[{self.dbname} Setup]: Creating Auto User Schema(s)')
            if self.config.userconfigs:
                for name, schema_config in self.config.userconfigs.items():
                    schema = LazyUserSchema.get_schema(schema_config, is_dev=self.config.is_dev)
                    self._db[name] = LazyDBModel(name, schema, LazyUserSchema.get_hash_schema(), is_dev=self.config.is_dev)
                    self._base_schemas[name] = self._db[name].schema
                    logger.info(f'[{self.dbname} Setup]: Created Custom User Schema: {name}')
            else:
                schema = LazyUserSchema.get_schema(is_dev=self.config.is_dev)
                self._db['user'] = LazyDBModel('user', schema, LazyUserSchema.get_hash_schema(), is_dev=self.config.is_dev)
                self._base_schemas['user'] = self._db['user'].schema
                logger.info(f'[{self.dbname} Setup]: Created Default User Schema')
        for name, schema in self.config.dbschema.items():
            hashschema = self.config.hashschema.get(name, None) if self.config.hashschema else None
            self._db[name] = LazyDBModel(name, schema, hashschema, is_dev=self.config.is_dev)
            self._base_schemas[name] = self._db[name].schema
            logger.info(f'[{self.dbname} Setup]: Created DB Schema Added for: {name}')

    def migrate_db(self):
        if not self.config.seeddata:
            logger.info(f'[{self.dbname} Migrate]: No Seed Data Provided. Skipping Migration')
        for schema_name, index in self._db.items():
            if index.idx == 0 and self.config.seeddata.get(schema_name):
                logger.info(f'[DB Migrate]: Running Migration for {schema_name}')
                for item in self.config.seeddata[schema_name]:
                    i = self._db[schema_name].create(data=item)
                    logger.info(f'[{self.dbname} Migrate]: Created Item [{schema_name}] = ID: {i.uid}, DBID: {i.dbid}')
        logger.info(f'[{self.dbname} Migrate]: Completed all Setup and Migration Tasks')
        self.save_db()

    def save_db(self):
        self.metrics.last_save = tstamp()
        self.metrics.num_saved += 1
        self.metrics.time_alive = self._alivetime.ablstime
        dbdata = {'db': self._db, 'timer': self._alivetime, 'metrics': self.metrics}
        self.cache.save(dbdata)

    def finalize_init(self):
        self.db = LazyObject(self._db)
        for name, index in self._db.items():
            schema_name = f'{name}_schema'
            setattr(self, schema_name, self._base_schemas[name])
            for method_type in ['get', 'create', 'remove', 'update']:
                method_name = f'{name}_{method_type}'
                method_func = lambda method_index=name, method_type=method_type, *args, **kwargs: self._db[method_index](*args, method=method_type, **kwargs)
                setattr(self, method_name, method_func)
        self.env = LazyEnv
        if self.config.autosave:
            if self.env.is_threadsafe:
                self.start_background()
            else:
                logger.warn(f'[{self.dbname} Finalize]: Currently not threadsafe. Call .start_background in main thread.')

    @property
    def dbname(self):
        return self.__class__.__name__

    def validate_cls(self, clsname):
        return self._db.get(clsname, None)

    def get(self, clsname, *args, **kwargs):
        if not self.validate_cls(clsname):
            return None
        return self._db[clsname].get(*args, **kwargs)

    def create(self, clsname, *args, **kwargs):
        if not self.validate_cls(clsname):
            return None
        return self._db[clsname].create(*args, **kwargs)

    def update(self, clsname, *args, **kwargs):
        if not self.validate_cls(clsname):
            return None
        return self._db[clsname].update(*args, **kwargs)

    def remove(self, clsname, *args, **kwargs):
        if not self.validate_cls(clsname):
            return None
        return self._db[clsname].remove(*args, **kwargs)

    def start_background(self):
        self.env.enable_watcher()
        self.t = threading.Thread(target=self.background, daemon=True)
        self.t.start()
        self.env.add_thread(self.t)

    def background(self):
        logger.info(f'[{self.dbname} Background]: DB AutoSaver Active. Saving Every: {self.config.savefreq} secs')
        microsleep = self.config.savefreq / 20
        while self.alive:
            for _ in range(20):
                time.sleep(microsleep)
                if self.env.killed:
                    self.alive = False
                    break
            self.save_db()
            if not self.alive:
                break

    def __call__(self, clsname, method, uid: int=None, dbid: str=None, *args, **kwargs):
        func = getattr(self, method)
        return func(self, clsname, *args, uid=uid, dbid=dbid, **kwargs)