import contextlib
from sqlalchemy import func, text
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer, get_sqlmodel_dict
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
class AsyncORMType(object):
    """
    Only contains async methods
    """
    id: str = Column(Text, default=create_unique_id, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=create_timestamp, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default=create_timestamp, server_default=func.now(), onupdate=create_timestamp)

    @classmethod
    def _filter(cls, query: Optional[Select]=None, **kwargs) -> Select:
        """
        Build a filter query
        """
        query = query if query is not None else select(cls)
        return query.where(*[getattr(cls, key) == value for key, value in kwargs.items()])

    @classmethod
    def _build_query(cls, query: Optional[Select]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, **kwargs) -> Select:
        """
        Build a query
        """
        query = query if query is not None else select(cls)
        query = query.where(*[getattr(cls, key) == value for key, value in kwargs.items()])
        if load_attrs:
            load_attr_method = get_attr_func(load_attr_method)
            for attr in load_attrs:
                query = query.options(load_attr_method(getattr(cls, attr)))
        return query

    def _filter_update_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Filter update data
        """
        data = {}
        for field, value in kwargs.items():
            if not hasattr(self, field):
                continue
            if getattr(self, field) == value:
                continue
            data[field] = value
        return data or None

    @classmethod
    def _handle_exception(cls, msg: Optional[str]=None, e: Optional[Exception]=None, verbose: Optional[bool]=False):
        """
        Handle exception
        """
        msg = msg or f'{cls.__name__} not found'
        if verbose:
            logger.trace(msg, error=e)
        raise e or HTTPException(status_code=404, detail=msg)

    @classmethod
    async def create(cls, **kwargs) -> Type['AsyncORMType']:
        """
        Create a new instance of the model
        """
        async with PostgresDB.async_session() as db_sess:
            new_cls = cls(**kwargs)
            db_sess.add(new_cls)
            await db_sess.flush()
            await db_sess.commit()
            await db_sess.refresh(new_cls)
        return new_cls

    @classmethod
    def _create(cls, **kwargs) -> Type['AsyncORMType']:
        """
        Create a new instance of the model
        """
        with PostgresDB.session() as db_sess:
            new_cls = cls(**kwargs)
            db_sess.add(new_cls)
            db_sess.flush()
            db_sess.commit()
            db_sess.refresh(new_cls)
        return new_cls

    async def refresh(self, session: Optional[AsyncSession]=None, **kwargs):
        """
        Refresh a record
        """
        async with PostgresDB.async_session() as db_sess:
            local_object = await db_sess.merge(self)
            db_sess.add(local_object)
            await db_sess.commit()
            await db_sess.refresh(local_object)
            self = local_object

    async def _update(self, **kwargs):
        """
        Update a record
        """
        async with PostgresDB.async_session() as db_sess:
            query = sqlalchemy_update(self.__class__).where(self.__class__.id == self.id).values(**kwargs)
            await db_sess.execute(query)
            await db_sess.commit()
        self = await self.get(id=self.id)

    async def update_inplace(self, **kwargs):
        """
        Update a record inplace
        """
        async with PostgresDB.async_session() as db_sess:
            local_object = await db_sess.merge(self)
            db_sess.add(local_object)
            await db_sess.refresh(local_object)
            await db_sess.commit()
            await db_sess.flush()
            self = local_object
        return self

    async def save(self, **kwargs):
        """
        Save a record
        """
        async with PostgresDB.async_session() as db_sess:
            local_object = await db_sess.merge(self)
            db_sess.add(local_object)
            await db_sess.refresh(local_object)
            await db_sess.commit()
        self = local_object
        return self

    async def update(self, **kwargs) -> Type['AsyncORMType']:
        """
        Update a record
        """
        filtered_update = self._filter_update_data(**kwargs)
        if not filtered_update:
            return self
        async with PostgresDB.async_session() as db_sess:
            for field, value in filtered_update.items():
                setattr(self, field, value)
            local_object = await db_sess.merge(self)
            db_sess.add(local_object)
            await db_sess.refresh(local_object)
            await db_sess.commit()
        self = local_object
        return self

    @classmethod
    async def get(cls, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, readonly: Optional[bool]=False, raise_exceptions: Optional[bool]=True, _model_type: Optional[ModelType]=None, _verbose: Optional[bool]=False, **kwargs) -> Type['AsyncORMType']:
        """
        Get a record
        """
        async with PostgresDB.async_session(ro=readonly) as db_sess:
            query = cls._build_query(load_attrs=load_attrs, load_attr_method=load_attr_method, **kwargs)
            result = (await db_sess.execute(query)).scalar_one_or_none()
            if not result and raise_exceptions:
                cls._handle_exception(e=NoResultFound(), verbose=_verbose)
            if result is not None and _model_type is not None:
                result = cast(_model_type, result)
        return result

    @classmethod
    def _get(cls, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, readonly: Optional[bool]=False, raise_exceptions: Optional[bool]=True, _model_type: Optional[ModelType]=None, _verbose: Optional[bool]=False, **kwargs) -> Type['AsyncORMType']:
        """
        Get a record
        """
        with PostgresDB.session(ro=readonly) as db_sess:
            query = cls._build_query(load_attrs=load_attrs, load_attr_method=load_attr_method, **kwargs)
            result = db_sess.execute(query).scalar_one_or_none()
            if not result and raise_exceptions:
                cls._handle_exception(e=NoResultFound(), verbose=_verbose)
            if result is not None and _model_type is not None:
                result = cast(_model_type, result)
        return result

    @classmethod
    async def get_all(cls, skip: Optional[int]=None, limit: Optional[int]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, readonly: Optional[bool]=False, raise_exceptions: Optional[bool]=True, _model_type: Optional[ModelType]=None, **kwargs) -> List[Type['AsyncORMType']]:
        """
        Get all records
        """
        async with PostgresDB.async_session(ro=readonly) as db_sess:
            query = cls._build_query(load_attrs=load_attrs, load_attr_method=load_attr_method, **kwargs)
            if skip is not None:
                query = query.offset(skip)
            if limit is not None:
                query = query.limit(limit)
            results = (await db_sess.scalars(query)).all()
            if not results and raise_exceptions:
                cls._handle_exception(e=NoResultFound())
            if results is not None and _model_type is not None:
                results = cast(List[_model_type], results)
        return results

    @classmethod
    async def first(cls, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, readonly: Optional[bool]=False, raise_exceptions: Optional[bool]=True, _model_type: Optional[ModelType]=None, **kwargs) -> Optional[Union[Type['AsyncORMType'], ModelType]]:
        """
        Return the first result of a query.
        """
        async with PostgresDB.async_session(ro=readonly) as db_sess:
            query = cls._build_query(load_attrs=load_attrs, load_attr_method=load_attr_method, **kwargs)
            result = (await db_sess.scalars(query)).first()
            if not result and raise_exceptions:
                cls._handle_exception(e=NoResultFound())
            if result is not None and _model_type is not None:
                result = cast(_model_type, result)
        return result

    @classmethod
    async def delete(cls, **kwargs) -> Type['AsyncORMType']:
        """
        Delete a record
        """
        obj = await cls.get(**kwargs, raise_exceptions=True)
        async with PostgresDB.async_session() as db_sess:
            db_sess.delete(obj)
            await db_sess.commit()
        return obj

    @classmethod
    async def delete_all(cls):
        async with PostgresDB.async_session() as db_sess:
            query = sqlalchemy_delete(cls)
            await db_sess.execute(query)
            await db_sess.commit()

    @classmethod
    async def exists(cls, **kwargs) -> bool:
        """
        Return True if a record exists
        """
        async with PostgresDB.async_session(ro=True) as db_sess:
            query = cls._filter(sqlalchemy_exists(), **kwargs).select()
            res = await db_sess.scalar(query)
        return res

    @classmethod
    async def count(cls, readonly: Optional[bool]=True, **kwargs) -> int:
        """
        Count the number of records in the table.
        """
        query = cls._filter(**kwargs) if kwargs else select(func.count(cls.id))
        try:
            async with PostgresDB.async_session(ro=readonly) as db_sess:
                return await db_sess.scalar(query)
        except Exception as e:
            logger.error(e)
            return 0

    def dict(self, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None, safe_encode: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Return a dictionary representation of the model.
        """
        data = self.pydantic_model.dict(exclude=exclude, include=include, **kwargs)
        if safe_encode:
            data = {key: object_serializer(value) for key, value in data.items()}
        return data

    def json(self, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None, exclude_none: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Return a dictionary representation of the model.
        """
        return self.pydantic_model.json(exclude=exclude, include=include, exclude_none=exclude_none, **kwargs)

    def diff(self, other: Union[Any, 'AsyncORMType']) -> Dict[str, Any]:
        """
        Return a dictionary of the differences between this model and another.
        """
        return {key: value for key, value in self.dict().items() if key in other.dict() and self.dict()[key] != other.dict()[key]}

    @lazyproperty
    def pydantic_model(self) -> Type[BaseModel]:
        """
        Return the Pydantic model for this ORM model.
        """
        return get_pydantic_model(self)

    @classmethod
    async def get_or_create(cls, filterby: Optional[Iterable[str]]=None, **kwargs) -> Tuple[Type['AsyncORMType'], bool]:
        """
        Create a new instance of the model, or return the existing one.
        """
        filterby = [list(kwargs.keys())[0]] if filterby is None else filterby
        _filterby = {key: kwargs.get(key) for key in filterby}
        result = await cls.get(**_filterby, raise_exceptions=False, _verbose=False)
        if result is not None:
            return (result, False)
        return (await cls.create(**kwargs), True)

    @classmethod
    def register(cls, filterby: Optional[Iterable[str]]=None, **kwargs) -> Tuple[Type['AsyncORMType'], bool]:
        """
        Create a new instance of the model, or return the existing one.

        Non-async version of `get_or_create` for use in synchronous code.
        """
        filterby = [list(kwargs.keys())[0]] if filterby is None else filterby
        _filterby = {key: kwargs.get(key) for key in filterby}
        result = cls._get(**_filterby, raise_exceptions=False, _verbose=False)
        return (result, False) if result is not None else (cls._create(**kwargs), True)

    @classmethod
    async def get_or_none(cls, **kwargs) -> Optional[Type['AsyncORMType']]:
        """
        Return an instance of the model, or None.
        """
        return await cls.get(**kwargs, raise_exceptions=False, _verbose=False)

    @classmethod
    async def get_or_create_or_update(cls, filterby: Optional[Iterable[str]]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, **kwargs: Dict) -> Tuple[Type['AsyncORMType'], bool]:
        """
        Create a new instance of the model, 
        or return the existing one after updating it.

        filterby: A list of fields to filter by when checking for an existing record.

        Returns a tuple of (instance, created | updated), where created is a boolean
        """
        filterby = [list(kwargs.keys())[0]] if filterby is None else filterby
        _filterby = {key: kwargs.get(key) for key in filterby}
        result = await cls.get(**_filterby, raise_exceptions=False, _verbose=False, load_attrs=load_attrs, load_attr_method=load_attr_method)
        if result is None:
            new_cls = await cls.create(**kwargs)
            return (new_cls, True)
        if (update_data := result._filter_update_data(**kwargs)):
            return (await result.update(**update_data), True)
        return (result, False)

    @classmethod
    async def create_or_update(cls, filterby: Optional[Iterable[str]]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, **kwargs) -> Tuple[Type['AsyncORMType'], bool]:
        """
        Create a new instance of the model, 
        or return the existing one after updating it.

        Returns a tuple of (instance, (true: created or updated, false: not updated), where created is a boolean
        """
        filterby = [list(kwargs.keys())[0]] if filterby is None else filterby
        _filterby = {key: kwargs.get(key) for key in filterby}
        result = await cls.get(**_filterby, raise_exceptions=False, _verbose=False, load_attrs=load_attrs, load_attr_method=load_attr_method)
        if result is None:
            return (await cls.create(**kwargs), True)
        if (update_data := result._filter_update_data(**kwargs)):
            logger.info(f'Updating {result} with {update_data}')
            return (await result.update(**update_data), True)
        return (result, False)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self.dict()})>'