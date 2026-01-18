import logging
from datetime import datetime as dt
from typing import Dict, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.amadeus.base import AmadeusBaseTool
class AmadeusFlightSearch(AmadeusBaseTool):
    """Tool for searching for a single flight between two airports."""
    name: str = 'single_flight_search'
    description: str = ' Use this tool to search for a single flight between the origin and  destination airports at a departure between an earliest and  latest datetime. '
    args_schema: Type[FlightSearchSchema] = FlightSearchSchema

    def _run(self, originLocationCode: str, destinationLocationCode: str, departureDateTimeEarliest: str, departureDateTimeLatest: str, page_number: int=1, run_manager: Optional[CallbackManagerForToolRun]=None) -> list:
        try:
            from amadeus import ResponseError
        except ImportError as e:
            raise ImportError('Unable to import amadeus, please install with `pip install amadeus`.') from e
        RESULTS_PER_PAGE = 10
        client = self.client
        earliestDeparture = dt.strptime(departureDateTimeEarliest, '%Y-%m-%dT%H:%M:%S')
        latestDeparture = dt.strptime(departureDateTimeLatest, '%Y-%m-%dT%H:%M:%S')
        if earliestDeparture.date() != latestDeparture.date():
            logger.error(" Error: Earliest and latest departure dates need to be the  same date. If you're trying to search for round-trip  flights, call this function for the outbound flight first,  and then call again for the return flight. ")
            return [None]
        response = None
        try:
            response = client.shopping.flight_offers_search.get(originLocationCode=originLocationCode, destinationLocationCode=destinationLocationCode, departureDate=latestDeparture.strftime('%Y-%m-%d'), adults=1)
        except ResponseError as error:
            print(error)
        output = []
        if response is not None:
            for offer in response.data:
                itinerary: Dict = {}
                itinerary['price'] = {}
                itinerary['price']['total'] = offer['price']['total']
                currency = offer['price']['currency']
                currency = response.result['dictionaries']['currencies'][currency]
                itinerary['price']['currency'] = {}
                itinerary['price']['currency'] = currency
                segments = []
                for segment in offer['itineraries'][0]['segments']:
                    flight = {}
                    flight['departure'] = segment['departure']
                    flight['arrival'] = segment['arrival']
                    flight['flightNumber'] = segment['number']
                    carrier = segment['carrierCode']
                    carrier = response.result['dictionaries']['carriers'][carrier]
                    flight['carrier'] = carrier
                    segments.append(flight)
                itinerary['segments'] = []
                itinerary['segments'] = segments
                output.append(itinerary)
        for index, offer in enumerate(output):
            offerDeparture = dt.strptime(offer['segments'][0]['departure']['at'], '%Y-%m-%dT%H:%M:%S')
            if offerDeparture > latestDeparture:
                output.pop(index)
        startIndex = (page_number - 1) * RESULTS_PER_PAGE
        endIndex = startIndex + RESULTS_PER_PAGE
        return output[startIndex:endIndex]