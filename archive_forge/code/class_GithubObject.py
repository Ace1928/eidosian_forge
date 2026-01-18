import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
class GithubObject:
    """
    Base class for all classes representing objects returned by the API.
    """
    '\n    A global debug flag to enable header validation by requester for all objects\n    '
    CHECK_AFTER_INIT_FLAG = False
    _url: Attribute[str]

    @classmethod
    def setCheckAfterInitFlag(cls, flag: bool) -> None:
        cls.CHECK_AFTER_INIT_FLAG = flag

    def __init__(self, requester: 'Requester', headers: Dict[str, Union[str, int]], attributes: Any, completed: bool):
        self._requester = requester
        self._initAttributes()
        self._storeAndUseAttributes(headers, attributes)
        if self.CHECK_AFTER_INIT_FLAG:
            requester.check_me(self)

    def _storeAndUseAttributes(self, headers: Dict[str, Union[str, int]], attributes: Any) -> None:
        self._headers = headers
        self._rawData = attributes
        self._useAttributes(attributes)

    @property
    def raw_data(self) -> Dict[str, Any]:
        """
        :type: dict
        """
        self._completeIfNeeded()
        return self._rawData

    @property
    def raw_headers(self) -> Dict[str, Union[str, int]]:
        """
        :type: dict
        """
        self._completeIfNeeded()
        return self._headers

    @staticmethod
    def _parentUrl(url: str) -> str:
        return '/'.join(url.split('/')[:-1])

    @staticmethod
    def __makeSimpleAttribute(value: Any, type: Type[T]) -> Attribute[T]:
        if value is None or isinstance(value, type):
            return _ValuedAttribute(value)
        else:
            return _BadAttribute(value, type)

    @staticmethod
    def __makeSimpleListAttribute(value: list, type: Type[T]) -> Attribute[T]:
        if isinstance(value, list) and all((isinstance(element, type) for element in value)):
            return _ValuedAttribute(value)
        else:
            return _BadAttribute(value, [type])

    @staticmethod
    def __makeTransformedAttribute(value: T, type: Type[T], transform: Callable[[T], K]) -> Attribute[K]:
        if value is None:
            return _ValuedAttribute(None)
        elif isinstance(value, type):
            try:
                return _ValuedAttribute(transform(value))
            except Exception as e:
                return _BadAttribute(value, type, e)
        else:
            return _BadAttribute(value, type)

    @staticmethod
    def _makeStringAttribute(value: Optional[Union[int, str]]) -> Attribute[str]:
        return GithubObject.__makeSimpleAttribute(value, str)

    @staticmethod
    def _makeIntAttribute(value: Optional[Union[int, str]]) -> Attribute[int]:
        return GithubObject.__makeSimpleAttribute(value, int)

    @staticmethod
    def _makeDecimalAttribute(value: Optional[Decimal]) -> Attribute[Decimal]:
        return GithubObject.__makeSimpleAttribute(value, Decimal)

    @staticmethod
    def _makeFloatAttribute(value: Optional[float]) -> Attribute[float]:
        return GithubObject.__makeSimpleAttribute(value, float)

    @staticmethod
    def _makeBoolAttribute(value: Optional[bool]) -> Attribute[bool]:
        return GithubObject.__makeSimpleAttribute(value, bool)

    @staticmethod
    def _makeDictAttribute(value: Dict[str, Any]) -> Attribute[Dict[str, Any]]:
        return GithubObject.__makeSimpleAttribute(value, dict)

    @staticmethod
    def _makeTimestampAttribute(value: int) -> Attribute[datetime]:
        return GithubObject.__makeTransformedAttribute(value, int, lambda t: datetime.fromtimestamp(t, tz=timezone.utc))

    @staticmethod
    def _makeDatetimeAttribute(value: Optional[str]) -> Attribute[datetime]:
        return GithubObject.__makeTransformedAttribute(value, str, _datetime_from_github_isoformat)

    @staticmethod
    def _makeHttpDatetimeAttribute(value: Optional[str]) -> Attribute[datetime]:
        return GithubObject.__makeTransformedAttribute(value, str, _datetime_from_http_date)

    def _makeClassAttribute(self, klass: Type[T_gh], value: Any) -> Attribute[T_gh]:
        return GithubObject.__makeTransformedAttribute(value, dict, lambda value: klass(self._requester, self._headers, value, completed=False))

    @staticmethod
    def _makeListOfStringsAttribute(value: Union[List[List[str]], List[str], List[Union[str, int]]]) -> Attribute:
        return GithubObject.__makeSimpleListAttribute(value, str)

    @staticmethod
    def _makeListOfIntsAttribute(value: List[int]) -> Attribute:
        return GithubObject.__makeSimpleListAttribute(value, int)

    @staticmethod
    def _makeListOfDictsAttribute(value: List[Dict[str, Union[str, List[Dict[str, Union[str, List[int]]]]]]]) -> Attribute:
        return GithubObject.__makeSimpleListAttribute(value, dict)

    @staticmethod
    def _makeListOfListOfStringsAttribute(value: List[List[str]]) -> Attribute:
        return GithubObject.__makeSimpleListAttribute(value, list)

    def _makeListOfClassesAttribute(self, klass: Type[T_gh], value: Any) -> Attribute[List[T_gh]]:
        if isinstance(value, list) and all((isinstance(element, dict) for element in value)):
            return _ValuedAttribute([klass(self._requester, self._headers, element, completed=False) for element in value])
        else:
            return _BadAttribute(value, [dict])

    def _makeDictOfStringsToClassesAttribute(self, klass: Type[T_gh], value: Dict[str, Union[int, Dict[str, Union[str, int, None]], Dict[str, Union[str, int]]]]) -> Attribute[Dict[str, T_gh]]:
        if isinstance(value, dict) and all((isinstance(key, str) and isinstance(element, dict) for key, element in value.items())):
            return _ValuedAttribute({key: klass(self._requester, self._headers, element, completed=False) for key, element in value.items()})
        else:
            return _BadAttribute(value, {str: dict})

    @property
    def etag(self) -> Optional[str]:
        """
        :type: str
        """
        return self._headers.get(Consts.RES_ETAG)

    @property
    def last_modified(self) -> Optional[str]:
        """
        :type: str
        """
        return self._headers.get(Consts.RES_LAST_MODIFIED)

    @property
    def last_modified_datetime(self) -> Optional[datetime]:
        """
        :type: datetime
        """
        return self._makeHttpDatetimeAttribute(self.last_modified).value

    def get__repr__(self, params: Dict[str, Any]) -> str:
        """
        Converts the object to a nicely printable string.
        """

        def format_params(params: Dict[str, Any]) -> typing.Generator[str, None, None]:
            items = list(params.items())
            for k, v in sorted(items, key=itemgetter(0), reverse=True):
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                if isinstance(v, str):
                    v = f'"{v}"'
                yield f'{k}={v}'
        return '{class_name}({params})'.format(class_name=self.__class__.__name__, params=', '.join(list(format_params(params))))

    def _initAttributes(self) -> None:
        raise NotImplementedError('BUG: Not Implemented _initAttributes')

    def _useAttributes(self, attributes: Any) -> None:
        raise NotImplementedError('BUG: Not Implemented _useAttributes')

    def _completeIfNeeded(self) -> None:
        raise NotImplementedError('BUG: Not Implemented _completeIfNeeded')