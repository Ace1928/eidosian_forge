from ._base import *
class GDELTFilters:

    def __init__(self, start_date: Optional[str]=None, end_date: Optional[str]=None, timespan: Optional[str]=None, num_records: int=250, keyword: Optional[Filter]=None, domain: Optional[Filter]=None, domain_exact: Optional[Filter]=None, near: Optional[str]=None, repeat: Optional[str]=None, country: Optional[Filter]=None, theme: Optional[Filter]=None) -> None:
        self.query_params: List[str] = []
        self._valid_countries: List[str] = []
        self._valid_themes: List[str] = []
        if not start_date and (not end_date) and (not timespan):
            raise ValueError('Must provide either start_date and end_date, or timespan')
        if start_date and end_date and timespan:
            raise ValueError('Can only provide either start_date and end_date, or timespan')
        if keyword:
            self.query_params.append(self._keyword_to_string(keyword))
        if domain:
            self.query_params.append(self._filter_to_string('domain', domain))
        if domain_exact:
            self.query_params.append(self._filter_to_string('domainis', domain_exact))
        if country:
            self.query_params.append(self._filter_to_string('sourcecountry', country))
        if theme:
            self.query_params.append(self._filter_to_string('theme', theme))
        if near:
            self.query_params.append(near)
        if repeat:
            self.query_params.append(repeat)
        if start_date:
            self.query_params.append(f'&startdatetime={start_date.replace('-', '')}000000')
            self.query_params.append(f'&enddatetime={end_date.replace('-', '')}000000')
        else:
            self.query_params.append(f'&timespan={timespan}')
        if num_records > 250:
            raise ValueError(f'num_records must 250 or less, not {num_records}')
        self.query_params.append(f'&maxrecords={str(num_records)}')

    @property
    def query_string(self) -> str:
        return ''.join(self.query_params)

    @staticmethod
    def _filter_to_string(name: str, f: Filter) -> str:
        if type(f) == str:
            return f'{name}:{f}'
        else:
            return '(' + ' OR '.join((f'{name}:{clause}' for clause in f)) + ')'

    @staticmethod
    def _keyword_to_string(keywords: Filter) -> str:
        if type(keywords) == str:
            return f'"{keywords}"'
        else:
            return '(' + ' OR '.join((f'"{word}"' if ' ' in word else word for word in keywords)) + ')'