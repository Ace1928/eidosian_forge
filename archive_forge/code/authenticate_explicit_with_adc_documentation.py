from google.cloud import storage
import google.oauth2.credentials
import google.auth

    List storage buckets by authenticating with ADC.

    // TODO(Developer):
    //  1. Before running this sample,
    //  set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    //  2. Replace the project variable.
    //  3. Make sure you have the necessary permission to list storage buckets: "storage.buckets.list"
    