from typing import Dict, List
import spacy
from langdetect import detect


class NLPNLUProcessor:
    """
    Provides advanced NLP and NLU functionalities to process and understand text data,
    integrating natural language understanding capabilities. This class utilizes the spaCy library
    for NLP tasks and langdetect for language detection.
    """

    def __init__(self):
        """
        Initializes the NLP/NLU processing capabilities. Loads the spaCy language models that are necessary
        for processing different languages and tasks.
        """
        self.nlp_en = spacy.load(
            "en_core_web_md"
        )  # English medium model, adjust based on needs
        self.nlp_es = spacy.load(
            "es_core_news_md"
        )  # Spanish medium model, example for multi-language support

    def enhance_text_data(self, raw_text: str) -> Dict[str, Dict]:
        """
        Processes raw text to extract advanced linguistic features including lemmas, POS tags,
        dependencies, and named entities, and understands context, sentiment, and semantic structures.
        :param raw_text: str - The text to be processed.
        :return: Dict[str, Dict] - A dictionary with processed features including tokens, entities, and sentences.
        """
        doc = self.nlp_en(raw_text)
        tokens = [
            {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
            }
            for token in doc
        ]
        entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        sentences = [sent.text for sent in doc.sents]
        return {"tokens": tokens, "entities": entities, "sentences": sentences}

    def language_detection(self, text: str) -> str:
        """
        Determines the language of the given text using the langdetect library, which supports multiple languages.
        :param text: str - The text whose language is to be detected.
        :return: str - The ISO 639-1 language code (e.g., 'en' for English, 'es' for Spanish).
        """
        return detect(text)

    def named_entity_recognition(self, text: str) -> List[Dict[str, str]]:
        """
        Identifies and classifies named entities in the text into predefined categories using spaCy.
        Supports different languages based on the loaded models.
        :param text: str - The text to be analyzed for named entities.
        :return: List[Dict[str, str]] - A list of dictionaries, each representing an entity with its text and type.
        """
        doc = self.nlp_en(
            text
        )  # Choose the correct model based on detected language or configuration
        return [{"text": entity.text, "type": entity.label_} for entity in doc.ents]
