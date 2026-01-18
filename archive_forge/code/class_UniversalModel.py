from pydantic import BaseModel, create_model, ValidationError, Field
from typing import Dict, Type, Any, Tuple, Union, TypeVar, get_type_hints, Optional
class UniversalModel(BaseModel):
    """
    A base class for a universal model that can dynamically evolve over time.
    This class serves as the foundation for creating models on-the-fly based on runtime data,
    allowing for the dynamic extension and adaptation of data models in response to changing requirements.
    """

    @classmethod
    def create_dynamic_model(cls, name: str, fields: FieldsDictType) -> ModelType:
        """
        Dynamically creates a new model extending the current one with additional fields.

        Args:
            name (str): The name of the new model class.
            fields (FieldsDictType): A dictionary where keys are field names and values are tuples of (type, default value).

        Returns:
            ModelType: A new Pydantic model class with the specified fields added.

        Raises:
            ValueError: If any of the new fields clash with existing fields in the base model.
        """
        existing_fields = set(cls.model_fields.keys())
        conflicting_fields = existing_fields.intersection(fields.keys())
        if conflicting_fields:
            raise ValueError(f'Fields {conflicting_fields} already exist in the base model.')
        model_fields = {field: (ftype, Field(default=default)) if default is not None else (ftype, Field(...)) for field, (ftype, default) in fields.items()}
        new_model = create_model(name, __base__=cls, **model_fields)
        return new_model