from openlineage.client.run import (
    RunEvent,
    RunState,
    Run,
    Job,
    Dataset,
    OutputDataset,
    InputDataset,
)
from openlineage.client.client import OpenLineageClient, OpenLineageClientOptions
from openlineage.client.facet import (
    SqlJobFacet,
    SchemaDatasetFacet,
    SchemaField,
    OutputStatisticsOutputDatasetFacet,
    SourceCodeLocationJobFacet,
    NominalTimeRunFacet,
    DataQualityMetricsInputDatasetFacet,
    ColumnMetric,
)
import uuid
from datetime import datetime, timezone, timedelta
import time
from random import random

# Create an OpenLineage client instance
client = OpenLineageClient(url="http://localhost:5000")

# Define the input dataset with its schema
input_dataset_name = "input_data.csv"
input_dataset_facet = SchemaDatasetFacet(
    fields=[
        SchemaField(name="id", type="integer"),
        SchemaField(name="name", type="string"),
    ]
)

# Define the output dataset with its schema
output_dataset_name = "output_data.csv"
output_dataset_facet = SchemaDatasetFacet(
    fields=[
        SchemaField(name="id", type="integer"),
        SchemaField(name="name", type="string"),
        SchemaField(name="processed_flag", type="boolean"),
    ]
)

# Start a new job run with the defined datasets
job_run_id = uuid.uuid4()
job = Job(
    namespace="default",
    name="data_processing_job",
    inputs=[
        InputDataset(
            namespace="default",
            name=input_dataset_name,
            facets={"schema": input_dataset_facet},
        )
    ],
    outputs=[
        OutputDataset(
            namespace="default",
            name=output_dataset_name,
            facets={"schema": output_dataset_facet},
        )
    ],
)
run = Run(
    runId=str(job_run_id),
    facets={"nominalTime": NominalTimeRunFacet(datetime.now(timezone.utc))},
)
client.emit(RunEvent(eventType=RunState.START, run=run, job=job))

# Perform data processing steps
# Placeholder for data processing logic

# End the job run
client.emit(RunEvent(eventType=RunState.COMPLETE, run=run, job=job))
